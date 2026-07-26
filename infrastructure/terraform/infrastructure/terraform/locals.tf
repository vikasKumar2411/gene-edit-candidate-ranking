locals {
  project_name = "gene-edit-ranking"

  state_machine_name = "${local.project_name}-batch-workflow"

  step_functions_role_name = "StepFunctionsGeneEditRankingRole"

  sagemaker_execution_role_name = (
    "AmazonSageMakerExecutionRole-GeneEditRanking"
  )

  log_group_name = (
    "/aws/vendedlogs/states/${local.state_machine_name}"
  )

  dashboard_name = "${local.project_name}-operations"
}
