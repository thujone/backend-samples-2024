import { fork } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import { v4 as uuidV4 } from 'uuid';
import { encode } from 'gpt-tokenizer';
import { ModerationCreateResponse } from 'openai/resources/moderations.js';
import { ChatCompletion } from 'openai/resources/chat/completions.js';
import { BackoffOptions, backOff } from 'exponential-backoff';
import { httpStatusCodes } from '../../../routes/utils.js';
import { convertIdToString, convertDateToIsoString } from '../utils.js';
import logger from '../../../utils/logger.js';
import {
  PGComment,
  GraphQLCommentRequest,
  commentValidator,
  messageValidator,
} from '../models/PGComment.js';
import { PGModerationResult } from '../models/PGModerationResult.js';
import { CommentOpenAiClient } from '../clients/CommentOpenAiClient.js';
import { PostgresClientError } from '../clients/PostgresClient.js';
import {
  CommentPostgresClient,
  CommentResponse,
  CommentsResetResponse,
} from '../clients/CommentPostgresClient.js';
import { Comments } from '../../../../types/DataExchangePerson.js';
import {
  ModerationDetails,
  ModerationMessage,
  PGModerationResultWithStatus,
} from './ModerationWorker.js';

export enum COMMENT_STATUS_ENUM {
  SUCCESS = 'success',
  ERROR = 'error',
  PENDING = 'pending',
}

/**
 * Provides services for creating, updating, and moderating comments using both
 * OpenAI's API for moderations and completions, and PostgreSQL for storage.
 */
export class CommentService {
  private _openAiClient: CommentOpenAiClient;
  private _postgresClient: CommentPostgresClient;
  private _promptTokenUnitCostPerThousand: number = 0.0015;
  private _completionTokenUnitCostPerThousand: number = 0.004;
  private _backoffOptions: BackoffOptions = {
    jitter: 'none',
    numOfAttempts: 5,
    startingDelay: 20e3, // 20 seconds
    maxDelay: 60e3, // 60 seconds
    timeMultiple: 1.5,
  };
  private _calculateNextDelay(
    startingDelay: number,
    timeMultiple: number,
    attemptNumber: number,
    maxDelay: number,
  ): number {
    return Math.min(startingDelay * Math.pow(timeMultiple, attemptNumber), maxDelay) / 1000;
  }

  /**
   * Initializes a new instance of the CommentService with dependencies.
   *
   * @param openAiClient Instance of CommentOpenAiClient for handling requests to the OpenAI API.
   * @param postgresClient Instance of CommentPostgresClient for handling database operations.
   */
  constructor(openAiClient: CommentOpenAiClient, postgresClient: CommentPostgresClient) {
    this._openAiClient = openAiClient;
    this._postgresClient = postgresClient;
  }

  /**
   * Validates a comment object against a predefined schema.
   * @param comment The comment object to validate.
   * @throws Throws an error if the validation fails.
   */
  async validateComment(comment: PGComment): Promise<void> {
    const { error } = commentValidator.validate(comment, { abortEarly: false });
    if (error) {
      const errorDetails = error.details.map((detail) => detail.message).join(', ');
      logger.error('Validation error in comment:', { details: errorDetails, comment });
      throw new Error(`Validation error in comment: ${errorDetails}`);
    }
  }

  /**
   * Gets a comment by its ID. Serializes the ID to a string before querying the database.
   * @param commentId - Unique identifier of the comment which can be a string or number.
   * @returns A promise that resolves to the comment if found, or null if not found.
   * @throws Throws an error if the database operation fails.
   */
  async getComment(commentId: string | number): Promise<PGComment | null> {
    const serializedCommentId: string = convertIdToString(commentId);
    try {
      return await this._postgresClient.getComment(serializedCommentId);
    } catch (error: unknown) {
      logger.error(
        {
          event: 'GetCommentError',
          error: error instanceof Error ? { message: error.message, stack: error.stack } : error,
        },
        'Failed to retrieve comment.',
      );
      throw new Error('Failed to retrieve comment.');
    }
  }

  /**
   * Gets a list of comments by its report ID. Converts PGComment objects to DataExchangePerson Comment
   * @param reportId - Unique identifier of the report.
   * @param reportType - Type of report. ex: person, phone, email
   * @param brandSlug - Brand slug of the report.
   * @returns A promise that resolves to a list of comments if found, or null if not found.
   * @throws Throws an error if the database operation fails.
   */
  async getCommentsByReport(
    reportId: string,
    reportType: string,
    brandSlug: string,
  ): Promise<Comments | null> {
    try {
      const pgComments = await this._postgresClient.getCommentsByReport(
        reportId,
        reportType,
        brandSlug,
      );
      if (!pgComments) return null;

      const comments = {} as Comments;
      comments.comments = [];
      pgComments.forEach((cm) => {
        comments.comments.push({
          id: cm.commentId,
          report_id: cm.reportId,
          report_type: cm.reportType ?? 'person',
          section: cm.section ?? 'general',
          message: cm.message,
          customer_id: cm.customerId ?? '',
          moderation_status: cm.moderationStatus,
          brand_slug: cm.brandSlug,
          verified_report_owner: cm.verifiedReportOwner ?? false,
          customer_first_name: cm.customerFirstName ?? '',
          customer_last_name: cm.customerLastName ?? '',
          created: cm.created ? convertDateToIsoString(cm.created) : '',
          updated: cm.updated ? convertDateToIsoString(cm.updated) : '',
        });
      });
      comments.total_available = comments.comments.length;

      return comments;
    } catch (error) {
      logger.error(
        {
          event: 'getCommentsByReportError',
          error: error instanceof Error ? { message: error.message, stack: error.stack } : error,
        },
        'Failed to retrieve comments by report ID.',
      );
      throw new Error('Failed to retrieve comments by report ID.');
    }
  }

  /**
   * Processes a new comment by storing it and moderating it using the OpenAI API.
   *
   * It validates the PGComment object then stores the comment in the database. Then it processes
   * the moderation status of the comment using the OpenAI API.
   *
   * The moderation status is determined in two stages: initial moderation and chat completion. The initial
   * moderation call checks for flaggable content and length, while the chat completion call returns a more
   * detailed moderation decision, with reasoning for the decision.
   *
   * The moderation is handled asynchronously by forking a child process to handle the moderation
   * ("ModerationWorker"). If the OpenAI API is unreachable, the moderation is retried using exponential
   * backoff. In such cases, the child process messages the parent process (i.e., this method) using
   * interprocess communication (IPC) to indicate that the moderation is in retry mode. Instead of waiting
   * for the OpenAI response, the parent is immediately notified with a 202 ACCEPTED status code. Assuming
   * the OpenAI API is reachable, the ModerationWorker makes the two OpenAI API calls and returns the
   * moderation result to the parent process, at which point the ModerationWorker process is terminated.
   *
   * @param commentRequest The commentRequest object to process or reprocess.
   * @returns A promise resolving to a CommentResponse.
   * @throws Throws an error if the database insertion fails (for new comments) or if one of the
   * moderation API calls fail.
   */
  async processNewComment(commentRequest: GraphQLCommentRequest): Promise<CommentResponse> {
    let commentResponse: CommentResponse;
    let comment = this._mapInsertRequestToPGComment(commentRequest, true);

    // Insert the comment into the database.
    try {
      await this.validateComment(comment); // Invalid comment will throw an error.
      commentResponse = await this._postgresClient.insertComment(comment);
      if (commentResponse.status === COMMENT_STATUS_ENUM.ERROR) {
        throw new Error('Comment was not inserted.');
      }
    } catch (error) {
      logger.error('Error processing comment:', error);
      throw error;
    }

    try {
      // Fork a child process with retry enabled.
      const moderationResponse: PGModerationResultWithStatus = await this._forkModerationWorker(
        comment,
        true,
      );

      // If the moderation is complete (either accepted or rejected), save the moderation result.
      if (moderationResponse.status === httpStatusCodes.Ok) {
        await this.processModerationResult(comment, moderationResponse);
        return this._buildResponse(comment);

        // Retry mode for the moderation worker. Return 202 ACCEPTED status along with the original
        // commentResponse as 'pending'.
      } else if (moderationResponse.status === httpStatusCodes.Accepted) {
        return this._buildResponse(comment);

        // If the moderation worker does not return a status, just return the original
        // commentResponse as 'pending'.
      } else {
        return commentResponse!;
      }
    } catch (error) {
      logger.error(
        {
          event: 'ModerationError',
          error: error instanceof Error ? { message: error.message, stack: error.stack } : error,
        },
        'Moderation error: OpenAI API calls failed.',
      );

      throw new Error('Moderation error: OpenAI API calls failed.');
    }
  }

  /**
   * Moderates a comment by forking a child process to handle the moderation task. Then, it listens
   * for messages from the child process to determine the moderation status, and to terminate the
   * child process once the moderation is complete.
   *
   * @param comment The comment to be moderated.
   * @param enableRetry A boolean indicating whether to enable retry logic for the moderation.
   *
   * @returns A promise that resolves with the moderation result including the status.
   */
  private async _forkModerationWorker(
    comment: PGComment,
    enableRetry: boolean,
  ): Promise<PGModerationResultWithStatus> {
    const __filename = fileURLToPath(import.meta.url);
    const __dirname = path.dirname(__filename);
    const moderationWorkerFilepath = path.join(__dirname, './ModerationWorker.js');

    return new Promise((resolve, reject) => {
      const forkedChildProcess = fork(moderationWorkerFilepath);

      // Send the comment to the moderation worker
      forkedChildProcess.send({ comment: JSON.stringify(comment), enableRetry: enableRetry });

      // Listen for a message from the moderation worker
      forkedChildProcess.on('message', async (workerMessage: ModerationMessage) => {
        logger.trace('Message received from ModerationWorker:', workerMessage);

        if (workerMessage.type === 'completed') {
          resolve({
            status: httpStatusCodes.Ok,
            ...workerMessage.result!,
          });
        } else if (workerMessage.type === 'retry') {
          resolve({
            status: httpStatusCodes.Accepted,
            ...workerMessage.result!,
          });
        } else if (workerMessage.type === 'error') {
          reject(new Error(workerMessage.error));
        } else {
          logger.trace(
            `Unexpected message from moderation worker for comment with commentId '${comment.commentId}': `,
            workerMessage,
          );
        }
      });

      forkedChildProcess.on('exit', (code) => {
        if (code !== 0) {
          reject(
            new Error(
              `ModerationWorker process for commentId '${comment.commentId}' exited with code ${code}`,
            ),
          );
        }
        logger.trace(
          `ModerationWorker process for commentId '${comment.commentId}' exited with code '${code}'.`,
        );
      });
    });
  }

  /**
   * Processes an existing comment, where the message has been edited and needs to be re-moderated.
   *
   * The application logic flow is somewhat different for existing comments. We don't want to update
   * the existing comment message unless it passes re-moderation. We'll still validate the comment,
   * because the message itself is new. If it passes validation, we re-moderate the message immediately.
   * If it passes moderation, at that point the moderation status and comment are updated in the database.
   * If it fails, neither is updated and the rejection response is returned to the client.
   *
   * Because this method necessitates an immediate response, we fork the child process with retry
   * turned off.
   *
   * @param commentRequest The commentRequest object to process or reprocess.
   * @returns A promise resolving to a CommentResponse.
   * @throws An error if the database insertion fails (for new comments) or if one of the
   * moderation API calls fail.
   */
  async processExistingComment(commentRequest: GraphQLCommentRequest): Promise<CommentResponse> {
    let comment = this._mapInsertRequestToPGComment(commentRequest, false);

    try {
      // Fork with retry turned off.
      const moderationResponse: PGModerationResultWithStatus = await this._forkModerationWorker(
        comment,
        false,
      );
      // If the moderationStatus is 'accepted', save the moderation result and return the comment response
      // for more processing.
      if (moderationResponse.moderationStatus === 'accepted') {
        const updatedComment = await this.processModerationResult(comment, moderationResponse);
        return this._buildResponse(updatedComment);

        // Else the moderation worker has rejected the comment with status code 200.
        // Status code of 202 ACCEPTED is not an option for existing comments because retry is disabled.
        // Don't overwrite the existing, good comment message or moderation status. Just return the
        // rejection response to the front end.
      } else {
        comment.moderationStatus = 'rejected';
        return this._buildResponse(comment);
      }
    } catch (error) {
      logger.error(
        {
          event: 'ModerationError',
          error: error instanceof Error ? { message: error.message, stack: error.stack } : error,
        },
        'Moderation error: OpenAI API calls failed.',
      );

      throw new Error('Moderation error: OpenAI API calls failed.');
    }
  }

  /**
   * Archives a comment by its ID.
   * Converts the comment ID to a string to ensure consistent handling and storage.
   *
   * @param commentId - Unique identifier of the comment.
   * @returns A promise that resolves to the `commentId` and `isArchived` status.
   * @throws Throws an error if the update operation fails or if the comment is not found.
   */
  async archiveComment(
    commentId: string | number,
  ): Promise<{ commentId: string; isArchived: boolean }> {
    const serializedCommentId = convertIdToString(commentId);
    try {
      return await this._postgresClient.archiveComment(serializedCommentId);
    } catch (error) {
      if (error instanceof PostgresClientError && error.statusCode === 404) {
        throw error; // Re-throw the error to be caught by the resolver
      }
      logger.error(
        {
          event: 'ArchiveCommentError',
          error: error instanceof Error ? { message: error.message, stack: error.stack } : error,
        },
        'Failed to archive comment.',
      );
      throw new Error('Failed to archive comment.');
    }
  }

  /**
   * Performs a "VRO reset" by archiving comments associated with a report and customer.
   *
   * @param reportId The unique identifier of the report.
   * @param customerId The unique identifier of the customer.
   * @param brandSlug The slug of the brand associated with the report.
   * @returns A promise that resolves to a CommentsResetResponse.
   */
  async resetComments(
    reportId: string | number,
    customerId: string | number,
    brandSlug: string,
  ): Promise<CommentsResetResponse> {
    const serializedReportId = convertIdToString(reportId);
    const serializedCustomerId = convertIdToString(customerId);
    try {
      const archivedCommentIds = await this._postgresClient.archiveComments(
        serializedReportId,
        serializedCustomerId,
        brandSlug,
      );

      if (!archivedCommentIds?.length) {
        return {
          status: COMMENT_STATUS_ENUM.SUCCESS,
          message: `No comments to reset.`,
          archivedCommentIds: [],
        };
      }
      return {
        status: COMMENT_STATUS_ENUM.SUCCESS,
        message: `Comments successfully archived.`,
        archivedCommentIds,
      };
    } catch (error) {
      logger.error(
        `Failed to reset comments.`,
        error,
        serializedReportId,
        serializedCustomerId,
        brandSlug,
      );
      return {
        status: COMMENT_STATUS_ENUM.ERROR,
        message: `Failed to reset comments for report ID '${serializedReportId}' and brand slug '${brandSlug}'.`,
        archivedCommentIds: null,
      };
    }
  }

  /**
   * Processes the moderation result by inserting the moderation response into the database and
   * updating the comment with the moderation decision and timestamp.
   *
   * @param comment - The comment object containing the details of the comment to be moderated.
   * @param moderationResponse - The moderation response object containing the results of the moderation.
   * @returns A promise that resolves to the updated `PGComment` object.
   * @throws Throws an error if the database operation fails, including a 400 error for duplicate
   * inserts.
   */
  async processModerationResult(
    comment: PGComment,
    moderationResponse: PGModerationResultWithStatus,
  ): Promise<PGComment> {
    try {
      await this._postgresClient.upsertModerationResult(moderationResponse, comment.commentId);
    } catch (error: unknown) {
      const errorInstance = error instanceof Error ? error : new Error(String(error));
      if (errorInstance instanceof PostgresClientError && errorInstance.statusCode === 400) {
        throw new Error(`400 Error: ${errorInstance.message}`);
      }
      throw errorInstance;
    }

    // Now update the comment with the moderation decision and timestamp.
    const updatedCommentResponse = await this.updateCommentModerationStatus(
      comment.commentId as string,
      moderationResponse.moderationStatus,
    );
    comment.moderationStatus = updatedCommentResponse.data!.moderationStatus;
    comment.updated = new Date(updatedCommentResponse.data!.updated);
    return comment;
  }

  /**
   * Updates the moderation status and 'updated' timestamp fields of an existing comment. This
   * method is called after the moderation process is complete and the moderation status is decided.
   *
   * @param commentId The ID of the comment to update.
   * @param moderationStatus The new moderation status to be set.
   * @returns A promise that resolves to the updated comment object, or null if the update fails.
   */
  async updateCommentModerationStatus(
    commentId: string,
    moderationStatus: string,
  ): Promise<CommentResponse> {
    return await this._postgresClient.updateCommentModerationStatus(commentId, moderationStatus);
  }

  /**
   * Performs the initial moderation on a comment to determine if it violates moderation standards
   * due to content or length. Uses exponential backoff for retrying API calls.
   *
   * @param comment - The comment object containing the message to be moderated.
   * @param messageId - An identifier for the message, used primarily for logging purposes.
   * @param enableRetry - A boolean indicating whether to enable retry logic for the moderation.
   * @returns A promise that resolves to a `PGModerationResult` object.
   *
   * @throws Throws an error if the moderation API call fails or if the response is undefined.
   */
  async moderateComment(
    comment: PGComment,
    messageId: string | number,
    enableRetry: boolean,
  ): Promise<PGModerationResult> {
    try {
      const { jitter, numOfAttempts, startingDelay, timeMultiple, maxDelay } = this._backoffOptions;
      let wasRetried = false;

      // Do the initial screening using retryable backoff
      const initialModeration: ModerationCreateResponse = await backOff(
        () => {
          // This is the actual API call to OpenAI's moderation model.
          return this._openAiClient.createModeration(comment.message);
        },
        {
          retry: (error, attemptNumber) => {
            // This whole block is just retry logic for the initial moderation API call. There's a very
            // similar block in generateChatCompletion() below for the chat completion API call.
            const statusCode = error.status || error.response?.status;
            const nextDelay = this._calculateNextDelay(
              startingDelay!,
              timeMultiple!,
              attemptNumber,
              maxDelay!,
            );

            // Retry any number over 404, assuming the comment is new and not being re-moderated.
            const shouldRetry = enableRetry && statusCode > httpStatusCodes.NotFound;

            // Check to make sure the status code is a retryable one
            if (!shouldRetry) {
              logger.error(
                {
                  event: 'InitialModerationStatusNotRetryable',
                  statusCode: statusCode,
                  error:
                    error instanceof Error ? { message: error.message, stack: error.stack } : error,
                },
                `Message '${messageId}' Moderations API call failed with a non-retryable status code.`,
              );

              // Exit the retry loop
              return false;
            }

            // If we're entering retry mode, send a message to the parent process (see comment.ts)
            // so the Koa server can respond with a 202 ACCEPTED status code without waiting for the
            // moderation to complete.
            if (shouldRetry && attemptNumber === 1) {
              wasRetried = true;
              const emptyModerationResult = this._prepareModerationResult({
                commentId: comment.commentId as string,
                allowed: false,
                reason: '',
                cost: 0,
                status: httpStatusCodes.Accepted,
                wasRetried: wasRetried,
              });
              process.send?.({ type: 'retry', emptyModerationResult });
            }

            logger.error(
              {
                event: 'InitialModerationRetryAttempt',
                attemptNumber: attemptNumber,
                statusCode: statusCode,
                willRetry: shouldRetry,
                nextDelay: `${nextDelay}s`,
              },
              `Message '${messageId}' Moderations API call failed. Retrying attempt ${attemptNumber}
          out of ${numOfAttempts}. Next retry in about ${nextDelay} seconds.`,
            );

            return shouldRetry;
          },
          jitter: jitter === 'none' ? 'none' : undefined,
          numOfAttempts,
          startingDelay,
          timeMultiple,
          maxDelay,
        },
      );

      logger.debug('Initial moderation API call completed.');

      // So the initial moderation uses OpenAI's moderator model to screen for obvious issues.
      // If it is flagged during the initial moderation, it gives a list of categories that failed.
      // If it passes, we just move on to the chat completion call.
      if (initialModeration === undefined) {
        throw new Error('initialModeration response is undefined.');
      } else if (!initialModeration.results || initialModeration.results.length === 0) {
        throw new Error('initialModeration results array is empty or undefined.');
      }

      const initialModerationResult = initialModeration.results[0];
      const userMessage = { role: 'user', content: comment.message };
      const systemPrompt = { role: 'system', content: this._openAiClient.prompt };

      // If the comment is flagged by initial moderation, return a ModerationResult with the
      // flagged categories as the reason, and allowed as false.
      if (initialModerationResult.flagged) {
        const flaggedCategories = Object.entries(initialModerationResult.categories)
          .filter(([, value]) => value)
          .map(([key]) => key);
        const notFlaggedCategories = Object.entries(initialModerationResult.categories)
          .filter(([, value]) => !value)
          .map(([key]) => key);

        logger.trace(
          {
            event: 'CommentFlagged',
            flaggedCategories: flaggedCategories,
            notFlaggedCategories: notFlaggedCategories,
            detail: `Comment '${comment.commentId}' was flagged by initial moderation.`,
          },
          'Comment was flagged by initial moderation.',
        );

        return this._prepareModerationResult({
          commentId: comment.commentId as string,
          allowed: false,
          reason: flaggedCategories.join(', '),
          cost: 0,
          status: httpStatusCodes.Ok,
          wasRetried: wasRetried,
        });

        // If the comment is too long, return a ModerationResult with `allowed` as false.
      } else if (!this._validateChatSize(userMessage.content, systemPrompt.content)) {
        logger.trace(
          {
            event: 'CommentTooLong',
            detail: `Comment '${comment.commentId}' is too long. (> ${this._openAiClient.tokenLimit} tokens)`,
          },
          `Comment is too long.`,
        );

        return this._prepareModerationResult({
          commentId: comment.commentId as string,
          allowed: false,
          reason: `Comment is too long.`,
          cost: 0,
          status: httpStatusCodes.PayloadTooLarge,
          wasRetried: wasRetried,
        });

        // If the comment is not flagged, return an empty ModerationResult and proceed to the
        // ChatCompletion API.
      } else {
        logger.info(
          {
            event: 'CommentPassedInitialModeration',
            detail: `Comment '${comment.commentId}' passed initial moderation without being flagged.`,
          },
          'Comment passed initial moderation without being flagged.',
        );

        return this._prepareModerationResult({
          commentId: comment.commentId as string,
          allowed: true,
          reason: '',
          cost: 0,
          status: httpStatusCodes.Ok,
          wasRetried: wasRetried,
        });
      }
    } catch (error) {
      logger.error(
        {
          event: 'InitialModerationError',
          error: error instanceof Error ? { message: error.message, stack: error.stack } : error,
        },
        'Initial Moderation API call failed.',
      );

      throw new Error('Moderation error: Initial Moderation API call failed.');
    }
  }

  /**
   * Generates a moderation decision by invoking the ChatCompletion model from OpenAI.
   * Uses retry with exponential backoff. It categorizes the comment based on the rules described
   * in the moderation prompt. If the ChatCompletion API response is successful, it parses the
   * response to determine if the content is allowed or not, along with reasons for the decision.
   *
   * @param comment - The comment to moderate.
   * @param messageId - Message ID, used for logging.
   * @param wasRetried - A boolean indicating whether the initial moderation was retried, which
   * would inform whether or not the moderation result still needs to be inserted into the database.
   * @param enableRetry - A boolean indicating whether to enable retry logic for the moderation.
   * @returns A promise that resolves to a `PGModerationResult` object containing the final moderation
   * decision and why.
   * @throws Throws an error if the ChatCompletion API call fails or if the response is undefined.
   */
  async generateChatCompletion(
    comment: PGComment,
    messageId: string | number,
    wasRetried: boolean,
    enableRetry: boolean,
  ): Promise<PGModerationResult> {
    try {
      const { jitter, numOfAttempts, startingDelay, timeMultiple, maxDelay } = this._backoffOptions;

      // Call the ChatCompletion API using retryable backoff
      const chatCompletionResponse: ChatCompletion = await backOff(
        () => {
          return this._openAiClient.createChatCompletion(comment.message);
        },
        {
          retry: (error, attemptNumber) => {
            const statusCode = error.status || error.response?.status;
            const nextDelay = this._calculateNextDelay(
              startingDelay!,
              timeMultiple!,
              attemptNumber,
              maxDelay!,
            );

            // Retry any number over 404, assuming the comment is new and not being re-moderated.
            const shouldRetry = enableRetry && statusCode > httpStatusCodes.NotFound;

            // Check to make sure the status code is a retryable one.
            // Keep in mind, a 202 could have already been sent by moderateComment() above.
            if (!shouldRetry) {
              logger.error(
                {
                  event: 'ChatCompletionStatusNotRetryable',
                  statusCode: statusCode,
                  error:
                    error instanceof Error ? { message: error.message, stack: error.stack } : error,
                },
                `Message '${messageId}' ChatCompletion API call failed with a non-retryable status code.`,
              );

              // Exit the retry loop
              return false;
            }

            // Tell the parent process not to wait any longer, because we're entering retry mode
            if (shouldRetry && attemptNumber === 1) {
              wasRetried = true;
              const emptyModerationResult = this._prepareModerationResult({
                commentId: comment.commentId as string,
                allowed: false,
                reason: '',
                cost: 0,
                status: httpStatusCodes.Accepted,
                wasRetried: wasRetried,
              });
              process.send?.({ type: 'retry', emptyModerationResult });
            }

            logger.error(
              {
                event: 'ChatCompletionRetryAttempt',
                attemptNumber: attemptNumber,
                statusCode: statusCode,
                willRetry: shouldRetry,
                nextDelay: `${nextDelay}s`,
                messageId: messageId,
                commentId: comment.commentId,
              },
              `Message '${messageId}' Chat Completion API call failed. Retrying attempt
            ${attemptNumber} out of ${numOfAttempts}. Next retry in about ${nextDelay} seconds.`,
            );
            return shouldRetry;
          },
          jitter: jitter === 'none' ? 'none' : undefined,
          numOfAttempts,
          startingDelay,
          timeMultiple,
          maxDelay,
        },
      );

      if (chatCompletionResponse === undefined) {
        throw new Error('ChatCompletion API response is undefined.');
      }

      if (!chatCompletionResponse.choices || chatCompletionResponse.choices.length === 0) {
        throw new Error('ChatCompletion API response has no choices.');
      }

      // The response is a json-encoded string, so we parse it to get the actual result.
      const completedMessage = JSON.parse(chatCompletionResponse.choices[0].message.content || '');
      const reason = completedMessage?.reason;
      const allowed = completedMessage?.allowed;
      const cost = this._calculateCost(
        chatCompletionResponse.usage?.completion_tokens || 0,
        chatCompletionResponse.usage?.prompt_tokens || 0,
      );

      logger.debug(
        {
          event: 'ChatCompletionSuccess',
          messageId: messageId,
          commentId: comment.commentId,
          allowed,
          reason,
          cost,
        },
        'Chat completion response received successfully.',
      );

      return this._prepareModerationResult({
        allowed,
        reason,
        cost,
        commentId: comment.commentId as string,
        status: httpStatusCodes.Ok,
        wasRetried,
      });
    } catch (error) {
      logger.error(
        {
          event: 'ChatCompletionError',
          messageId: messageId,
          commentId: comment.commentId,
          error: error instanceof Error ? { message: error.message, stack: error.stack } : error,
        },
        'Chat Completion API call failed.',
      );

      throw new Error(`ChatCompletionError: Chat Completion API`);
    }
  }

  /**
   * Edits an existing comment and re-moderates it.
   *
   * @param commentId The ID of the comment to edit.
   * @param message The new message to set for the comment.
   * @returns A promise that resolves to the updated CommentResponse.
   */
  async editCommentMessage(commentId: string | number, message: string): Promise<CommentResponse> {
    // Serialize the comment ID to a string.
    const serializedCommentId = convertIdToString(commentId);

    // Validate the message field
    const { error } = messageValidator.validate({ message });
    if (error) {
      const errorDetail = error.details[0].message;
      logger.error('Validation error in message:', { detail: errorDetail, message });
      return {
        status: COMMENT_STATUS_ENUM.ERROR,
        message: `Validation error in message: ${errorDetail}`,
        data: null,
      };
    }

    try {
      const comment = await this.getComment(serializedCommentId);
      if (!comment) {
        const errorMessage = 'No comment found for the given comment ID.';
        logger.error(errorMessage, commentId);
        throw new Error(errorMessage);
      }

      // Update just the message of the comment object
      comment.message = message;
      const graphqlCommentRequest = this._mapCommentToCommentRequest(comment);

      // Reprocess the updated comment (moderation and chat completion)
      const reprocessedComment = await this.processExistingComment(graphqlCommentRequest);

      // If the moderationStatus is 'accepted', save the moderation result and the new comment message.
      // Otherwise, return the rejection response to the front end without persisting any of it.
      if (reprocessedComment.data?.moderationStatus === 'accepted') {
        const editedProcessedComment = await this._postgresClient.editCommentMessage(
          reprocessedComment.data.commentId,
          reprocessedComment.data.message,
          reprocessedComment.data.moderationStatus,
        );
        return this._buildResponse(editedProcessedComment);
      } else {
        return reprocessedComment;
      }
    } catch (error) {
      if (error instanceof PostgresClientError && error.statusCode === 404) {
        logger.warn(`Comment with ID '${serializedCommentId}' not found.`);
        return {
          status: COMMENT_STATUS_ENUM.ERROR,
          message: `Comment with ID '${serializedCommentId}' not found.`,
          data: null,
        };
      }
      logger.error('Failed to edit comment message:', error);
      return {
        status: COMMENT_STATUS_ENUM.ERROR,
        message: 'Failed to edit comment message.',
        data: null,
      };
    }
  }

  /**
   * Maps a GraphQLCommentRequest object to a PGComment object for insertion.
   * @param comment The GraphQLCommentRequest object to map.
   * @returns PGComment The mapped PGComment object.
   */
  private _mapInsertRequestToPGComment(
    commentRequest: GraphQLCommentRequest,
    isNewComment: boolean,
  ): PGComment {
    return {
      commentId: isNewComment ? uuidV4() : convertIdToString(commentRequest.commentId),
      reportType: commentRequest.reportType ?? 'person',
      idType: commentRequest.reportType === 'person' ? 'tcg_id' : commentRequest.reportType,
      section: commentRequest.section ?? 'general',
      message: commentRequest.message,
      customerFirstName: commentRequest.customerFirstName,
      customerLastName: commentRequest.customerLastName,
      moderationStatus: COMMENT_STATUS_ENUM.PENDING,
      brandSlug: commentRequest.brandSlug,
      raw: JSON.stringify(commentRequest),
      isArchived: commentRequest.isArchived ?? false,
      verifiedReportOwner: commentRequest.verifiedReportOwner ?? false,
      reportId: convertIdToString(commentRequest.reportId),
      customerId: commentRequest.customerId ? convertIdToString(commentRequest.customerId) : null,
      created: commentRequest.created ? new Date(commentRequest.created) : new Date(),
      updated: commentRequest.updated ? new Date(commentRequest.updated) : new Date(),
    };
  }
  /**
   * Maps a PGComment object to a GraphQLCommentRequest object.
   *
   * @param comment The PGComment object to map.
   * @returns The mapped GraphQLCommentRequest object.
   */
  private _mapCommentToCommentRequest(comment: PGComment): GraphQLCommentRequest {
    return {
      commentId: convertIdToString(comment.commentId),
      reportType: comment.reportType,
      idType: comment.idType,
      section: comment.section,
      message: comment.message,
      customerId: comment.customerId ? convertIdToString(comment.customerId) : null,
      customerFirstName: comment.customerFirstName,
      customerLastName: comment.customerLastName,
      created: comment.created ? comment.created.toISOString() : null,
      updated: comment.updated ? comment.updated.toISOString() : null,
      moderationStatus: comment.moderationStatus,
      brandSlug: comment.brandSlug,
      reportId: convertIdToString(comment.reportId),
      isArchived: comment.isArchived,
      raw: comment.raw,
      verifiedReportOwner: comment.verifiedReportOwner,
    };
  }

  /**
   * Reset the moderation result to an initial state in the database before remoderating.
   *
   * @param updatedComment The comment data.
   * @returns Void.
   */
  private async _resetModerationResult(updatedComment: PGComment) {
    const initialModerationResult = this._prepareModerationResult({
      commentId: updatedComment.commentId as string,
      allowed: false,
      reason: '',
      cost: 0,
      status: httpStatusCodes.Ok,
      wasRetried: false,
    });

    try {
      await this._postgresClient.upsertModerationResult(
        initialModerationResult,
        updatedComment.commentId,
      );
    } catch (error) {
      logger.error('Error resetting moderation result:', error);
      throw new Error('Failed to reset moderation result.');
    }
  }

  /**
   * Builds a standard response for operations on comments.
   *
   * @param comment The comment data used for the response.
   * @returns An object containing relevant fields for the response.
   */
  private _buildResponse(comment: PGComment): CommentResponse {
    return {
      status: 'success',
      message: 'Comment processed successfully.',
      data: {
        reportType: comment.reportType ?? '',
        commentId: convertIdToString(comment.commentId),
        idType: comment.idType ?? '',
        section: comment.section ?? '',
        message: comment.message,
        customerId: comment.customerId ? convertIdToString(comment.customerId) : '',
        customerFirstName: comment.customerFirstName ?? '',
        customerLastName: comment.customerLastName ?? '',
        created: comment.created ? convertDateToIsoString(comment.created) : '',
        updated: comment.updated ? convertDateToIsoString(comment.updated) : '',
        moderationStatus: comment.moderationStatus ?? '',
        brandSlug: comment.brandSlug ?? '',
        reportId: convertIdToString(comment.reportId),
        isArchived: comment.isArchived,
        raw: comment.raw ?? '',
        verifiedReportOwner: comment.verifiedReportOwner,
      },
    };
  }

  /**
   * Prepares a moderation result object for insertion into the database.
   *
   * @param details - An object containing the moderation details including the commentId, allowed
   * status, reason for moderation decision, cost of moderation, and other relevant data.
   * @returns A `PGModerationResult` object ready for database insertion.
   */
  private _prepareModerationResult(details: ModerationDetails): PGModerationResult {
    return {
      commentId: details.commentId,
      moderationStatus: details.allowed ? 'accepted' : 'rejected',
      reason: details.reason,
      cost: details.cost,
      timestamp: new Date(),
      wasRetried: details.wasRetried,
    };
  }

  /**
   * Validates the size of the chat message against the token limit.
   *
   * @param userChat - The chat message from the user to be moderated.
   * @param systemPrompt - The system's prompt message used for moderation.
   * @returns A boolean indicating whether the combined token count of the user chat and system
   * prompt is within the limit.
   */
  private _validateChatSize(userChat: string, systemPrompt: string): boolean {
    const userChatTokens = userChat ? encode(userChat) : '';
    const systemPromptTokens = systemPrompt ? encode(systemPrompt) : '';

    logger.debug(
      `Completion tokens: ${userChatTokens.length}. System prompt tokens: ${systemPromptTokens.length}`,
    );

    return userChatTokens.length + systemPromptTokens.length <= this._openAiClient.tokenLimit;
  }

  /**
   * Calculates the cost associated with chat completion based on the number of tokens used.
   *
   * @param completionTokens - The number of tokens generated during the chat completion.
   * @param promptTokens - The number of tokens used in the moderation prompt.
   * @returns The total cost of moderation formatted to six decimal places.
   */
  private _calculateCost(completionTokens: number, promptTokens: number): number {
    try {
      const promptCost = (promptTokens / 1000) * this._promptTokenUnitCostPerThousand;
      const responseCost = (completionTokens / 1000) * this._completionTokenUnitCostPerThousand;
      const totalCost = promptCost + responseCost;
      return Number(totalCost.toFixed(6));
    } catch (error) {
      logger.error(`Error calculating cost: ${error}`);
      return 0;
    }
  }
}
