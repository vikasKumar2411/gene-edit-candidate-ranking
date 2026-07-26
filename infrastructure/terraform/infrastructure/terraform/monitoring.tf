resource "aws_cloudwatch_log_group" "step_functions" {
  name              = local.log_group_name
  retention_in_days = 14
}

resource "aws_cloudwatch_metric_alarm" "step_functions_failures" {
  alarm_name        = "gene-edit-ranking-step-functions-failures"
  alarm_description = "Alarm when the gene-edit ranking Step Functions workflow fails."

  namespace   = "AWS/States"
  metric_name = "ExecutionsFailed"

  dimensions = {
    StateMachineArn = "arn:aws:states:${var.aws_region}:${var.aws_account_id}:stateMachine:${local.state_machine_name}"
  }

  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
}

resource "aws_cloudwatch_metric_alarm" "processing_job_failures" {
  alarm_name        = "gene-edit-ranking-processing-job-failures"
  alarm_description = "Alarm when a SageMaker Processing job fails."

  namespace   = "AWS/SageMaker"
  metric_name = "ProcessingJobsFailed"

  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
}

resource "aws_cloudwatch_dashboard" "operations" {
  dashboard_name = local.dashboard_name

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6

        properties = {
          title  = "Step Functions Executions"
          region = var.aws_region
          view   = "timeSeries"
          stat   = "Sum"
          period = 300

          metrics = [
            [
              "AWS/States",
              "ExecutionsSucceeded",
              "StateMachineArn",
              "arn:aws:states:${var.aws_region}:${var.aws_account_id}:stateMachine:${local.state_machine_name}"
            ],
            [
              ".",
              "ExecutionsFailed",
              ".",
              "."
            ],
            [
              ".",
              "ExecutionsStarted",
              ".",
              "."
            ]
          ]
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6

        properties = {
          title  = "Step Functions Duration"
          region = var.aws_region
          view   = "timeSeries"
          stat   = "Average"
          period = 300

          metrics = [
            [
              "AWS/States",
              "ExecutionTime",
              "StateMachineArn",
              "arn:aws:states:${var.aws_region}:${var.aws_account_id}:stateMachine:${local.state_machine_name}"
            ]
          ]
        }
      },
      {
        type   = "alarm"
        x      = 0
        y      = 6
        width  = 24
        height = 6

        properties = {
          title = "Workflow Alarms"

          alarms = [
            "arn:aws:cloudwatch:${var.aws_region}:${var.aws_account_id}:alarm:gene-edit-ranking-step-functions-failures",
            "arn:aws:cloudwatch:${var.aws_region}:${var.aws_account_id}:alarm:gene-edit-ranking-processing-job-failures"
          ]
        }
      }
    ]
  })
}
