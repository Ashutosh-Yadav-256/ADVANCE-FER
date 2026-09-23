terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# --- Networking ---
resource "aws_vpc" "fer_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags = {
    Name        = "${var.app_name}-vpc"
    Environment = var.environment
  }
}

resource "aws_subnet" "public_1" {
  vpc_id            = aws_vpc.fer_vpc.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = "${var.aws_region}a"
  map_public_ip_on_launch = true
}

resource "aws_subnet" "public_2" {
  vpc_id            = aws_vpc.fer_vpc.id
  cidr_block        = "10.0.2.0/24"
  availability_zone = "${var.aws_region}b"
  map_public_ip_on_launch = true
}

# --- Database (PostgreSQL RDS) ---
resource "aws_db_subnet_group" "fer_db_subnets" {
  name       = "${var.app_name}-db-subnets"
  subnet_ids = [aws_subnet.public_1.id, aws_subnet.public_2.id]
}

resource "aws_db_instance" "fer_postgres" {
  identifier           = "${var.app_name}-postgres"
  engine               = "postgres"
  engine_version       = "16.1"
  instance_class       = "db.t4g.micro"
  allocated_storage    = 20
  db_name              = "fer_db"
  username             = "postgres"
  password             = var.db_password
  db_subnet_group_name = aws_db_subnet_group.fer_db_subnets.name
  skip_final_snapshot  = true
}

# --- Caching (ElastiCache Redis) ---
resource "aws_elasticache_cluster" "fer_redis" {
  cluster_id           = "${var.app_name}-redis"
  engine               = "redis"
  node_type            = "cache.t4g.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
}

# --- Container Orchestration (AWS ECS Fargate) ---
resource "aws_ecs_cluster" "fer_cluster" {
  name = "${var.app_name}-cluster"
}

resource "aws_ecs_task_definition" "fer_task" {
  family                   = "${var.app_name}-task"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024"
  memory                   = "2048"

  container_definitions = jsonencode([
    {
      name      = "fer-api"
      image     = "advance-fer:latest"
      essential = true
      portMappings = [
        {
          containerPort = var.container_port
          hostPort      = var.container_port
        }
      ]
      environment = [
        { name = "ENVIRONMENT", value = var.environment },
        { name = "DATABASE_URL", value = "postgresql://${aws_db_instance.fer_postgres.username}:${var.db_password}@${aws_db_instance.fer_postgres.endpoint}/fer_db" },
        { name = "REDIS_URL", value = "redis://${aws_elasticache_cluster.fer_redis.cache_nodes[0].address}:6379/0" }
      ]
    }
  ])
}

# --- Application Load Balancer ---
resource "aws_lb" "fer_alb" {
  name               = "${var.app_name}-alb"
  internal           = false
  load_balancer_type = "application"
  subnets            = [aws_subnet.public_1.id, aws_subnet.public_2.id]
}

resource "aws_lb_target_group" "fer_tg" {
  name        = "${var.app_name}-tg"
  port        = var.container_port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.fer_vpc.id
  target_type = "ip"

  health_check {
    path                = "/healthz"
    matcher             = "200"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "fer_listener" {
  load_balancer_arn = aws_lb.fer_alb.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.fer_tg.arn
  }
}
