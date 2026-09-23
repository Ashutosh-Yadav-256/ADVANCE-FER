output "alb_dns_name" {
  description = "Public DNS name of the Application Load Balancer"
  value       = aws_lb.fer_alb.dns_name
}

output "database_endpoint" {
  description = "RDS PostgreSQL host connection endpoint"
  value       = aws_db_instance.fer_postgres.endpoint
}

output "redis_endpoint" {
  description = "ElastiCache Redis primary cache node endpoint"
  value       = aws_elasticache_cluster.fer_redis.cache_nodes[0].address
}
