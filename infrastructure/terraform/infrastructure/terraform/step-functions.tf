resource "aws_sfn_state_machine" "batch_ranking" {
  name     = local.state_machine_name
  role_arn = aws_iam_role.step_functions.arn
  type     = "STANDARD"

  definition = jsonencode({
    Comment = "Run managed gene-edit candidate batch ranking"
    StartAt = "RunBatchRanking"

    States = {
      RunBatchRanking = {
        Type     = "Task"
        Resource = "arn:aws:states:::sagemaker:createProcessingJob.sync"

        Parameters = {
          "ProcessingJobName.$" = "$.processing_job_name"

          RoleArn = "arn:aws:iam::${var.aws_account_id}:role/${local.sagemaker_execution_role_name}"

          AppSpecification = {
            ImageUri = var.batch_image_uri

            "ContainerArguments.$" = "States.Array('--input-s3-uri', $.input_s3_uri, '--scoring-date', $.scoring_date)"
          }

          ProcessingResources = {
            ClusterConfig = {
              InstanceCount  = 1
              InstanceType   = "ml.m5.large"
              VolumeSizeInGB = 10
            }
          }

          StoppingCondition = {
            MaxRuntimeInSeconds = 1800
          }
        }

        ResultPath = "$.processing_result"

        Retry = [
          {
            ErrorEquals = [
              "SageMaker.AmazonSageMakerException",
              "SageMaker.ResourceLimitExceededException",
              "States.Timeout"
            ]

            IntervalSeconds = 30
            MaxAttempts     = 2
            BackoffRate     = 2
          }
        ]

        Catch = [
          {
            ErrorEquals = ["States.ALL"]
            ResultPath  = "$.error"
            Next        = "BatchRankingFailed"
          }
        ]

        Next = "BatchRankingSucceeded"
      }

      BatchRankingSucceeded = {
        Type = "Pass"

        Result = {
          status = "SUCCEEDED"
        }

        ResultPath = "$.workflow_status"
        End        = true
      }

      BatchRankingFailed = {
        Type  = "Fail"
        Error = "BatchRankingFailed"
        Cause = "The SageMaker batch-ranking processing job failed."
      }
    }
  })

  logging_configuration {
    include_execution_data = true
    level                  = "ALL"
    log_destination        = "${aws_cloudwatch_log_group.step_functions.arn}:*"
  }

  depends_on = [
    aws_iam_role_policy.run_processing_job,
    aws_iam_role_policy.cloudwatch_logging
  ]
}
