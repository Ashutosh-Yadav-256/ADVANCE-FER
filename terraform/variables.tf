variable "aws_region" {
  description = "AWS deployment region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment name"
  type        = string
  default     = "production"
}

variable "app_name" {
  description = "Application identifier"
  type        = string
  default     = "advance-fer"
}

variable "container_port" {
  description = "Container service listening port"
  type        = number
  default     = 8000
}

variable "db_password" {
  description = "PostgreSQL root administrator password"
  type        = string
  sensitive   = true
  default     = "ChangeMeStrongPassword123!"
}
