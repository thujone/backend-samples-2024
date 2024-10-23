import os
import csv
import requests
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.google.cloud.transfers.gcs_to_bigquery import GCSToBigQueryOperator
from airflow.providers.google.cloud.operators.dataflow import DataflowStartFlexTemplateOperator
from airflow.models import Variable
from datetime import datetime, timedelta

# DAG for downloading FTC DNC complaints, processing them, loading them into BigQuery, and
# loading BigQuery data into Bigtable.

# The following system variables can be set in Airflow > Admin > Variables
#
# bigquery_table - The name of the BigQuery table, defaults to 'ftc_do_not_call_violations'.
# bigquery_to_bigtable_jobname - The name of the Dataflow job that runs the BigQuery-to-Bigtable transformation,
#     defaults to 'ftc-do-not-call-violations-v4-prod'.
# ftc_dnc_path - The partial directory path containing the FTC CSV files, relative to the bucket, defaults to
#     'ftc/dnc'.
# tmp_dir - The local filepath where CSV files are downloaded from the FTC website, defaults to '/tmp'.

def download_file(**kwargs):
    """
    Downloads the CSV file for the given date from the FTC website.
    """
    date = kwargs['ds']
    url = f'https://www.ftc.gov/sites/default/files/DNC_Complaint_Numbers_{date}.csv'
    file = f'DNC_Complaint_Numbers_{date}.csv'
    tmp_dir = Variable.get('tmp_dir', default_var='/tmp')

    # Ensure the directory exists
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(tmp_dir, file), 'wb') as f:
            f.write(response.content)
    else:
        raise ValueError(f"Failed to download file: {response.status_code}")

    kwargs['ti'].xcom_push(key='file', value=file)

def add_filename_column(**kwargs):
    """
    Adds the Filename column to the CSV file.
    """
    file = kwargs['ti'].xcom_pull(task_ids='download_file', key='file')
    tmp_dir = Variable.get('tmp_dir', default_var='/tmp')
    filepath = os.path.join(tmp_dir, file)
    new_filepath = os.path.join(tmp_dir, f'processed_{file}')
    filename_col_value = file

    # Add the Filename column before writing the file to the bucket
    with open(filepath, 'r') as infile, open(new_filepath, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)
        header.append('Filename')
        writer.writerow(header)

        for row in reader:
            row.append(filename_col_value)
            writer.writerow(row)

    kwargs['ti'].xcom_push(key='new_filepath', value=new_filepath)

def upload_to_gcs(**kwargs):
    """
    Uploads the processed file to Google Cloud Storage.
    """
    from google.cloud import storage

    new_filepath = kwargs['ti'].xcom_pull(task_ids='add_filename_column', key='new_filepath')
    tmp_dir = Variable.get('tmp_dir', default_var='/tmp')
    ftc_dnc_path = Variable.get('ftc_dnc_path', default_var='ftc/dnc')
    bucket_name = 'tcg-data-files'
    file = kwargs['ti'].xcom_pull(task_ids='download_file', key='file')
    destination_blob_name = f'{ftc_dnc_path}/{file}'

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(new_filepath)

    print(f"File {new_filepath} uploaded to {bucket_name}/{destination_blob_name}.")
    kwargs['ti'].xcom_push(key='destination_blob_name', value=destination_blob_name)

def load_into_bigquery(**kwargs):
    """
    Loads the processed file from GCS into BigQuery.
    """
    bigquery_table = Variable.get('bigquery_table', default_var='ftc_do_not_call_violations')
    destination_blob_name = kwargs['ti'].xcom_pull(task_ids='upload_to_gcs', key='destination_blob_name')

    schema_fields = [
        {"name": "phone_number", "type": "STRING", "mode": "NULLABLE"},
        {"name": "created_date", "type": "STRING", "mode": "REQUIRED"},
        {"name": "violation_date", "type": "STRING", "mode": "REQUIRED"},
        {"name": "consumer_city", "type": "STRING", "mode": "NULLABLE"},
        {"name": "consumer_state", "type": "STRING", "mode": "NULLABLE"},
        {"name": "consumer_area_code", "type": "STRING", "mode": "NULLABLE"},
        {"name": "subject", "type": "STRING", "mode": "NULLABLE"},
        {"name": "recorded_message_or_robocall", "type": "STRING", "mode": "NULLABLE"},
        {"name": "filename", "type": "STRING", "mode": "NULLABLE"}
    ]

    load_to_bigquery_task = GCSToBigQueryOperator(
        task_id='load_to_bigquery',
        bucket='tcg-data-files',
        source_objects=[destination_blob_name],
        destination_project_dataset_table=f'tcg-data-exchange.source_data.{bigquery_table}',
        schema_fields=schema_fields,
        write_disposition='WRITE_APPEND',
        skip_leading_rows=1,
        field_delimiter=',',
        source_format='CSV',
        gcp_conn_id='google_cloud_default',
    )

    load_to_bigquery_task.execute(kwargs)

def load_into_bigtable(**kwargs):
    """
    Processes BigQuery data into Bigtable.
    """
    # For development, be sure to use 'ftc-do-not-call-violations-v4-dev-only'
    bigquery_table = Variable.get('bigquery_table', default_var='ftc_do_not_call_violations')
    bigquery_to_bigtable_jobname = Variable.get('bigquery_to_bigtable_jobname', default_var='ftc-do-not-call-violations-v4-prod')
    project_id = 'tcg-data-exchange'
    template_path = 'gs://dataflow-templates-us-east4/latest/flex/BigQuery_to_Bigtable'

    # Extract the filename from the download_file step
    filename = kwargs['ti'].xcom_pull(task_ids='download_file', key='file')

    dataflow_task = DataflowStartFlexTemplateOperator(
        task_id='load_into_bigtable',
        project_id=project_id,
        location='us-east4',
        body={
            'launchParameter': {
                'jobName': bigquery_to_bigtable_jobname,
                'containerSpecGcsPath': template_path,
                'parameters': {
                    'bigtableWriteProjectId': 'tcg-data',
                    'bigtableWriteInstanceId': 'vendors-mixed-instance',
                    'bigtableWriteTableId': 'ftc-do-not-call-violations',
                    'bigtableWriteColumnFamily': 'record',
                    'bigtableWriteAppProfile': 'default',
                    'inputTableSpec': f'tcg-data-exchange:source_data.{bigquery_table}',
                    'query': (
                        "SELECT "
                        "CONCAT(COALESCE(phone_number, ''), '#', COALESCE(created_date, '')) AS row_key, "
                        "COALESCE(phone_number, '') as phone_number, "
                        "COALESCE(created_date, '') as created_date, "
                        "COALESCE(violation_date, '') as violation_date, "
                        "COALESCE(consumer_city, '') as consumer_city, "
                        "COALESCE(consumer_state, '') as consumer_state, "
                        "COALESCE(consumer_area_code, '') as consumer_area_code, "
                        "COALESCE(subject, '') as subject, "
                        "COALESCE(recorded_message_or_robocall, '') as recorded_message_or_robocall, "
                        "COALESCE(filename, '') as filename "
                        f"FROM `tcg-data-exchange.source_data.{bigquery_table}` "
                        f"WHERE filename = '{filename}'"
                    ),
                    'queryLocation': 'US',
                    'readIdColumn': 'row_key'
                }
            }
        }
    )

    dataflow_task.execute(kwargs)

default_args = {
    'depends_on_past': False,
    'retries': 24,
    'retry_delay': timedelta(hours=1),
}
with DAG(
    'ftc_dnc_complaints',
    default_args=default_args,
    description='Downloads FTC DNC complaints and uploads them to GCS',
    start_date=datetime(2022, 5, 27),
    schedule_interval='0 0 * * 1-5',  # Runs at midnight every weekday
    catchup=True,
) as dag:

    download_task = PythonOperator(
        task_id='download_file',
        provide_context=True,
        python_callable=download_file
    )

    add_filename_column_task = PythonOperator(
        task_id='add_filename_column',
        provide_context=True,
        python_callable=add_filename_column
    )

    upload_task = PythonOperator(
        task_id='upload_to_gcs',
        provide_context=True,
        python_callable=upload_to_gcs
    )

    load_into_bigquery_task = PythonOperator(
        task_id='load_into_bigquery',
        provide_context=True,
        python_callable=load_into_bigquery
    )

    load_into_bigtable_task = PythonOperator(
        task_id='load_into_bigtable',
        provide_context=True,
        python_callable=load_into_bigtable
    )

    download_task >> add_filename_column_task >> upload_task >> load_into_bigquery_task >> load_into_bigtable_task
