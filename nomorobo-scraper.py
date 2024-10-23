from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.models import Variable
import requests
from collections import defaultdict
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import csv
import os
import re
from google.cloud import storage

# DAG for scraping robocaller phone numbers from Nomorobo, scraping the details (transcript, audio,
# etc.) for each number, writing the results to GCS, and subsequently loading the data into BigQuery
# and then Bigtable.
#
# The following system variables can be set in Airflow > Admin > Variables:
#
# nomorobo_numbers_path - The partial directory path containing the Nomorobo numbers CSV files,
#     relative to the bucket, defaults to 'nomorobo/numbers'.
# nomorobo_sequence_bookmark - The sequence of the three newest phone numbers in the Nomorobo stream,
#     defaults to an empty list.
# nomorobo_tmp_dir - The local filepath where scraped CSV files are written, defaults to '/tmp'.

def scrape_numbers():
    """
    Scrapes phone numbers from the Nomorobo website and stores them locally.
    """
    nomorobo_tmp_dir = Variable.get('nomorobo_tmp_dir', default_var='/tmp')
    nomorobo_sequence_bookmark = Variable.get('nomorobo_sequence_bookmark', default_var=[])

    url_template = "https://www.nomorobo.com/lookup?page={}"
    hourly_datetime = datetime.now().strftime("%Y-%m-%d-%H-00")
    file_name = os.path.join(nomorobo_tmp_dir, f"nomorobo-numbers-{hourly_datetime}.csv")

    if nomorobo_sequence_bookmark:
        nomorobo_sequence_bookmark = nomorobo_sequence_bookmark.split(',')
        print(f"Sequence bookmark to match: {' '.join(nomorobo_sequence_bookmark)}")
    else:
        print("No sequence bookmark found. Starting fresh.")

    all_new_numbers = []
    seen_numbers = set()
    number_count = defaultdict(int)
    page_number = 1
    max_pages = 1 if not nomorobo_sequence_bookmark else 15
    found_old_sequence = False
    previous_page_numbers = []

    while not found_old_sequence and page_number <= max_pages:
        url = url_template.format(page_number)
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        new_robocallers_section = soup.find('h5', string="New robocaller recordings")
        phone_numbers = []

        if new_robocallers_section:
            phone_number_div = new_robocallers_section.find_next('div', class_='row')
            phone_numbers = [a_tag.text.strip() for a_tag in phone_number_div.find_all('a')]

        if not phone_numbers:
            break

        print(f"Extracting page {page_number} into memory.")

        if previous_page_numbers:
            combined = previous_page_numbers[-2:] + phone_numbers[:2] if phone_numbers else []
            print(f"Looking for sequence bookmark across pages {page_number - 1} and {page_number}...")
            if combined == nomorobo_sequence_bookmark[:len(combined)]:
                print("Found the sequence bookmark across pages! Stopping scrape.")
                found_old_sequence = True
                break
            else:
                print("Sequence bookmark across pages not found. Continuing scrape.")

        print(f"Checking page {page_number} for sequence match...")

        sequence_position = -1
        for i in range(len(phone_numbers) - 2):
            if phone_numbers[i:i + 3] == nomorobo_sequence_bookmark:
                sequence_position = i + 1
                break

        if sequence_position != -1:
            if sequence_position == 1:
                print("No numbers to scrape.")
            else:
                for phone in phone_numbers[:sequence_position - 1]:
                    if phone not in seen_numbers:
                        seen_numbers.add(phone)
                        all_new_numbers.append(phone)
                print(f"Found the sequence bookmark on page {page_number} at position {sequence_position}. ")
                print(f"Scraped numbers from position 1 to {sequence_position - 1}. Stopping scrape.")

            found_old_sequence = True
            break

        for phone in phone_numbers:
            number_count[phone] += 1
            if phone not in seen_numbers:
                seen_numbers.add(phone)
                all_new_numbers.append(phone)

        previous_page_numbers = phone_numbers
        page_number += 1

    # Ensure we get three contiguous numbers for the sequence bookmark.
    response = requests.get(url_template.format(1))
    soup = BeautifulSoup(response.content, 'html.parser')
    new_robocallers_section = soup.find('h5', string="New robocaller recordings")
    first_page_numbers = []

    if new_robocallers_section:
        phone_number_div = new_robocallers_section.find_next('div', class_='row')
        first_page_numbers = [a_tag.text.strip() for a_tag in phone_number_div.find_all('a')]

    if all_new_numbers:
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            for phone in all_new_numbers:
                writer.writerow([phone])

        new_nomorobo_sequence_bookmark = first_page_numbers[:3]
        Variable.set('nomorobo_sequence_bookmark', ','.join(new_nomorobo_sequence_bookmark))

        print(f"New sequence bookmark: {' '.join(new_nomorobo_sequence_bookmark)}")
        print(f"Scraped and saved {len(all_new_numbers)} phone numbers to {file_name}")

        # Print out the dupes and how many times they appeared
        duplicates = {phone: count for phone, count in number_count.items() if count > 1}
        for phone, count in duplicates.items():
            print(f"Duplicate found: {phone} appeared {count} times.")

        return file_name
    else:
        print("No new numbers scraped. File will not be created.")
        return None

def upload_numbers_to_gcs(**kwargs):
    """
    Uploads the scraped numbers CSV file to Google Cloud Storage.
    """
    ti = kwargs['ti']
    file = ti.xcom_pull(task_ids='scrape_numbers_task')
    nomorobo_numbers_path = Variable.get('nomorobo_numbers_path', default_var='nomorobo/numbers')

    if file:
        upload_file_to_gcs = LocalFilesystemToGCSOperator(
            task_id='upload_numbers_to_gcs',
            src=file,
            dst=f'{nomorobo_numbers_path}/{os.path.basename(file)}',
            bucket='tcg-data-files',
            gcp_conn_id='google_cloud_default',
        )
        upload_file_to_gcs.execute(context=kwargs)
        ti.xcom_push(key='numbers_file', value=file)
    else:
        print("No file to upload.")

def scrape_details(**kwargs):
    # TODO: Add Airflow variables appropriately
    ti = kwargs['ti']
    file = ti.xcom_pull(task_ids='scrape_numbers_task')

    if not file:
        print("No numbers file to process.")
        return None
    else:
        print(f"Processing numbers file: {file}")
        nomorobo_tmp_dir = Variable.get('nomorobo_tmp_dir', default_var='/tmp')
        base_filename = os.path.basename(file)
        details_filename = base_filename.replace('numbers', 'details')
        details_file_path = os.path.join(nomorobo_tmp_dir, details_filename)

        with open(file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)
            total_details = len(rows)
            print(f"Total numbers to process: {total_details}")

        if total_details == 0:
            print("Numbers file is empty. No details to scrape.")
            return None

        with open(details_file_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['phone_number', 'date_blocked', 'transcript', 'audio_url', 'audio_filename'])

            count = 0

            for row in rows:
                count += 1
                phone_number_raw = row[0]
                phone_number = re.sub(r'\D', '', phone_number_raw)

                digits = re.sub(r'\D', '', phone_number_raw)
                if len(digits) == 10:
                    formatted_phone_number = '{}-{}-{}'.format(digits[:3], digits[3:6], digits[6:])
                else:
                    formatted_phone_number = phone_number_raw

                url = 'https://www.nomorobo.com/lookup/' + formatted_phone_number
                try:
                    response = requests.get(url, timeout=10)
                    response.raise_for_status()
                except Exception as e:
                    print(f"Failed to fetch details for {phone_number_raw}: {e}")
                    continue
                else:
                    soup = BeautifulSoup(response.content, 'html.parser')

                    audio_url = None
                    audio_tag = soup.find('audio', id='player')
                    if audio_tag:
                        source_tag = audio_tag.find('source')
                        if source_tag and source_tag.has_attr('src'):
                            audio_url = source_tag['src']

                    transcript = None
                    transcript_label = soup.find('strong', class_='title', string='Transcript')
                    if transcript_label:
                        transcript_content = transcript_label.find_next_sibling('span', class_='cont')
                        if transcript_content:
                            transcript_text = transcript_content.get_text(strip=True)
                            transcript = transcript_text

                    date_blocked = None
                    date_blocked_label = soup.find('strong', class_='title', string='Date Blocked')
                    if date_blocked_label:
                        date_blocked_content = date_blocked_label.find_next_sibling('span', class_='cont')
                        if date_blocked_content:
                            time_tag = date_blocked_content.find('time')
                            if time_tag and time_tag.has_attr('datetime'):
                                date_blocked_raw = time_tag['datetime']
                                date_blocked = date_blocked_raw + 'T00:00:00Z'

                    audio_filename = None
                    if audio_url:
                        audio_filename = os.path.basename(audio_url)
                        audio_filename = f"{os.path.splitext(audio_filename)[0]}.wav"
                    else:
                        audio_filename = None

                    writer.writerow([phone_number, date_blocked, transcript, audio_url, audio_filename])

                if count % 20 == 0 or count == total_details:
                    print(f"Scraped {count} out of {total_details} details pages.")

        print(f"Scraped details and saved to {details_file_path}")
        return details_file_path

def upload_details_to_gcs(**kwargs):
    """
    Uploads the scraped details CSV file to Google Cloud Storage.
    """
    ti = kwargs['ti']
    details_file = ti.xcom_pull(task_ids='scrape_details_task')
    nomorobo_details_path = 'nomorobo/number-details'

    if details_file:
        upload_file_to_gcs = LocalFilesystemToGCSOperator(
            task_id='upload_details_to_gcs',
            src=details_file,
            dst=f'{nomorobo_details_path}/{os.path.basename(details_file)}',
            bucket='tcg-data-files',
            gcp_conn_id='google_cloud_default',
        )
        upload_file_to_gcs.execute(context=kwargs)
    else:
        print("No details file to upload.")

def transfer_audio_files(**kwargs):
    ti = kwargs['ti']
    details_file = ti.xcom_pull(task_ids='scrape_details_task')

    if not details_file:
        print("No details file to process.")
        return None
    else:
        print(f"Processing details file: {details_file}")
        nomorobo_tmp_dir = Variable.get('nomorobo_tmp_dir', default_var='/tmp')
        audio_files_bucket_path = 'nomorobo/audio-files'

        with open(details_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            total_files = len(rows)
            print(f"Total audio files to process: {total_files}")

        storage_client = storage.Client()
        bucket_name = 'tcg-data-files'
        bucket = storage_client.bucket(bucket_name)

        for index, row in enumerate(rows):
            audio_url = row['audio_url']
            audio_filename = row['audio_filename']
            phone_number = row['phone_number']

            if audio_url and audio_filename:
                try:
                    response = requests.get(audio_url, stream=True, timeout=10)
                    response.raise_for_status()
                    local_audio_path = os.path.join(nomorobo_tmp_dir, audio_filename)
                    with open(local_audio_path, 'wb') as audio_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                audio_file.write(chunk)

                    blob = bucket.blob(f'{audio_files_bucket_path}/{audio_filename}')
                    blob.upload_from_filename(local_audio_path)

                    os.remove(local_audio_path)

                    print(f"Transferred audio file for {phone_number}: {audio_filename}")
                except Exception as e:
                    print(f"Failed to transfer audio file for {phone_number}: {e}")
            else:
                print(f"No audio URL for {phone_number}")
        return

default_args = {
    'depends_on_past': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=15),
}

with DAG(
    'nomorobo_scraper',
    default_args=default_args,
    description='Scrapes phone numbers from Nomorobo and uploads them to GCS',
    schedule_interval='0 * * * *',
    catchup=False,
    start_date=datetime(2024, 9, 13),
) as dag:

    scrape_numbers_task = PythonOperator(
        task_id='scrape_numbers_task',
        python_callable=scrape_numbers,
    )

    upload_numbers_to_gcs_task = PythonOperator(
        task_id='upload_numbers_to_gcs_task',
        python_callable=upload_numbers_to_gcs,
        provide_context=True,
    )

    scrape_details_task = PythonOperator(
        task_id='scrape_details_task',
        python_callable=scrape_details,
        provide_context=True,
    )

    upload_details_to_gcs_task = PythonOperator(
        task_id='upload_details_to_gcs_task',
        python_callable=upload_details_to_gcs,
        provide_context=True,
    )

    transfer_audio_files_task = PythonOperator(
        task_id='transfer_audio_files_task',
        python_callable=transfer_audio_files,
        provide_context=True,
    )

scrape_numbers_task >> upload_numbers_to_gcs_task >> scrape_details_task >> upload_details_to_gcs_task >> transfer_audio_files_task
