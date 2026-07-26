resource "aws_iam_role" "step_functions" {
  name = local.step_functions_role_name

  assume_role_policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Effect = "Allow"

        Principal = {
          Service = "states.amazonaws.com"
        }

        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "run_processing_job" {
  name = "RunGeneEditRankingProcessingJob"
  role = aws_iam_role.step_functions.name

  policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Sid    = "ManageSageMakerProcessingJobs"
        Effect = "Allow"

        Action = [
          "sagemaker:CreateProcessingJob",
          "sagemaker:DescribeProcessingJob",
          "sagemaker:StopProcessingJob",
          "sagemaker:AddTags"
        ]

        Resource = "*"
      },
      {
        Sid    = "PassSageMakerExecutionRole"
        Effect = "Allow"
        Action = "iam:PassRole"

        Resource = "arn:aws:iam::${var.aws_account_id}:role/${local.sagemaker_execution_role_name}"

        Condition = {
          StringEquals = {
            "iam:PassedToService" = "sagemaker.amazonaws.com"
          }
        }
      },
      {
        Sid    = "ManageStepFunctionEvents"
        Effect = "Allow"

        Action = [
          "events:PutTargets",
          "events:PutRule",
          "events:DescribeRule"
        ]

        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy" "cloudwatch_logging" {
  name = "StepFunctionsCloudWatchExecutionLogging"
  role = aws_iam_role.step_functions.name

  policy = jsonencode({
    Version = "2012-10-17"

    Statement = [
      {
        Sid    = "DeliverExecutionLogs"
        Effect = "Allow"

        Action = [
          "logs:CreateLogDelivery",
          "logs:CreateLogStream",
          "logs:GetLogDelivery",
          "logs:UpdateLogDelivery",
          "logs:DeleteLogDelivery",
          "logs:ListLogDeliveries",
          "logs:PutLogEvents",
          "logs:PutResourcePolicy",
          "logs:DescribeResourcePolicies",
          "logs:DescribeLogGroups"
        ]

        Resource = "*"
      }
    ]
  })
}
