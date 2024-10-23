import { PGComment } from '../models/PGComment.js';
import logger from '../../../utils/logger.js';
import { commentService } from '../index.js';
import { PGModerationResult } from '../models/PGModerationResult.js';

export interface ModerationDetails {
  commentId: string;
  allowed: boolean;
  reason: string;
  cost: number;
  status: number;
  wasRetried: boolean;
}

export interface PGModerationResultWithStatus extends PGModerationResult {
  status: number;
}

export interface ModerationMessage {
  type: 'completed' | 'retry' | 'error';
  result?: PGModerationResult;
  error?: string;
}

/**
 * ModerationWorker is always forked as a child process, but retry is enabled or disabled based on
 * whether the comment is brand new or is an existing comment being re-moderated.
 *
 * In the case of editing an existing comment message, the CommentService will instantiate the
 * ModerationWorker with retry capabilities turned off. This is because editing a comment necessitates
 * immediate re-moderation, and we must be able to return an 'accepted' or 'rejected' moderation
 * result immediately. If OpenAI or Postgres services are unavailable when an existing comment is
 * re-moderated, the moderation result will be 'rejected'.
 *
 * For new comments, the forked retry is enabled and is allowed to keep running separately from the
 * Koa process, which in turn allows Koa to respond to the client immediately with a 'pending'
 * moderation status, even if retry fails to process entirely. See forkModerationWorker() in
 * CommentService.ts for more details.
 */
export default class ModerationWorker {
  private _commentService = commentService;
  private _enableRetry: boolean;

  constructor(enableRetry: boolean) {
    this._commentService = commentService;
    this._enableRetry = enableRetry;
  }

  /**
   * Handles the moderation of comments by invoking moderation and chat completion services.
   * Results are communicated back to the parent process via inter-process communication.
   *
   * @param commentData JSON string of the comment data.
   */
  async moderate(commentData: string) {
    try {
      const comment: PGComment = JSON.parse(commentData);

      const initialModerationResult: PGModerationResult =
        await this._commentService.moderateComment(
          comment,
          comment.commentId.toString(),
          this._enableRetry,
        );
      let chatCompletionResult: PGModerationResult;
      let finalResult: PGModerationResultWithStatus;

      logger.info(
        {
          event: 'ModerationWorkerInitialSuccess',
          moderationResult: initialModerationResult,
        },
        `ModerationWorker has moderated initially with moderationStatus of '${initialModerationResult.moderationStatus}'.`,
      );

      // If the initial moderation result is 'accepted', generate chat completion and send the final
      // result to the parent process. (See forkModerationWorker in CommentService.ts)
      if (initialModerationResult.moderationStatus === 'accepted') {
        chatCompletionResult = await this._commentService.generateChatCompletion(
          comment,
          comment.commentId.toString(),
          initialModerationResult.wasRetried,
          this._enableRetry,
        );
        logger.info(
          {
            event: 'ModerationWorkerChatCompletionSuccess',
            chatCompletionResult: chatCompletionResult,
          },
          `ModerationWorker has finished chatCompletion with moderationStatus of '${chatCompletionResult.moderationStatus}'.`,
        );

        finalResult = { ...chatCompletionResult, status: 200 };

        // Else the initial moderation result is 'rejected', so skip the chat completion step and just
        // send back the initialModerationResult as the final result.
      } else {
        finalResult = { ...initialModerationResult, status: 200 };
      }

      // If at any point this comment went into retry mode for either the initial moderation
      // or chat completion, OR if the retry logic is disabled, that means the parent process is no
      // longer listening here, because 202 Accepted has already been returned to Koa / GraphQL OR
      // retry feature isn't even turned on. So instead of trying to send the final result back to
      // the parent process, we should just process the ModerationResult directly here.
      if (finalResult.wasRetried) {
        await commentService.processModerationResult(comment, finalResult);
      } else {
        const finalMessage = { type: 'completed', result: finalResult };
        process.send?.(finalMessage);
      }

      // Ensure the worker process exits after completing the task
      process.exit(0);
    } catch (error) {
      const errorMessage: ModerationMessage = {
        type: 'error',
        error: error instanceof Error ? error.message : 'Unexpected error occurred',
      };

      logger.error(
        {
          event:
            error instanceof Error ? 'ModerationWorkerError' : 'ModerationWorkerUnexpectedError',
          error: error,
        },
        `ModerationWorker has failed with error message '${errorMessage.error}'.`,
      );

      process.send?.(errorMessage);

      // Ensure the worker process exits even in case of an error
      process.exit(1);
    }
  }
}

// Have the ModerationWorker listen for IPC requests from the parent process.
// The message containing the comment data is sent from the _forkModerationWorker() method in
// CommentService.ts.
process.on('message', ({ comment, enableRetry }: { comment: string; enableRetry: boolean }) => {
  logger.info('ModerationWorker received a message!');
  const worker = new ModerationWorker(enableRetry);
  worker.moderate(comment);
});
