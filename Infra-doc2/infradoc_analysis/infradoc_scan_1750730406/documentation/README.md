# \

> **Full-Stack Web Service** - Web API service with background processing capabilities

## 🚀 Quick Start

```bash
# Start services
sudo systemctl start application-service

# Verify everything is running
curl http://localhost:8000/health

# View logs
tail -f /var/log/app/application.log
```

## 🎯 What This System Does


### Business Purpose


### Key Capabilities


## 🏗️ Architecture Overview


| Component | Details |
|-----------|---------|
| **Architecture Pattern** | Unknown |
| **Deployment Model** | Unknown |
| **Technology Stack** |  |
| **Processes Running** | 15 active processes |
| **Application Files** | 4 files analyzed |
| **Scalability** | Unknown |
| **Security Posture** | Unknown |
| **Operational Complexity** | Unknown |



## 🌐 API Reference

This service exposes **1 API endpoints** organized by functionality:

### Quick API Reference
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/generate-course` | Triggers the course generation process using an external service. |

**Base URL**: `http://localhost`

📚 **[Complete API Documentation →](./api_documentation.md)**



## 🏢 Business Logic

**Domain**: Unknown

### Core Business Functions

🏢 **[Detailed Business Logic Guide →](./business_logic.md)**


## 📁 Project Structure

```
└── \/
  └── opt/
    └── learnchain/
      ├── worker.py (Python) # This script acts as a worker that listens to an AW...
      ├── parsing_adapter.py (Python) # This script processes documents using the LlamaPar...
      ├── worker-2.py (Python) # This script acts as a worker that listens to an AW...
      └── create_course_knowledge_graph_neo.py (Python) # This script is designed to create a knowledge grap...
```

## ⚙️ Configuration


| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `LLAMA_CLOUD_API_KEY_2` | Used to authenticate requests to the LlamaParse service. | Yes | `None` |
| `DISABLE_IMG` | Overrides the --keep-images flag to disable image extraction. | No | `true` |
| `AWS_ACCESS_KEY_ID` | Used for authenticating with AWS S3 | Yes | `None` |
| `AWS_SECRET_ACCESS_KEY` | Used for authenticating with AWS S3 | Yes | `None` |
| `AWS_DEFAULT_REGION` | Specifies the AWS region for S3 operations | Yes | `us-east-2` |
| `OPENAI_API_KEY` | Used for authenticating with OpenAI's API | Yes | `None` |
| `NEO4J_URI` | Specifies the URI for connecting to the Neo4j database | Yes | `None` |
| `NEO4J_USER` | Username for Neo4j database authentication | Yes | `neo4j` |
| `NEO4J_PASSWORD` | Password for Neo4j database authentication | Yes | `None` |


## 🚦 Health & Monitoring


| Check | Endpoint/Command | Expected Response |
|-------|------------------|-------------------|
| **API Health** | `GET http://localhost/health` | `200 OK` |
| **/usr/bin/python3 Service** | `systemctl status /usr/bin/python3` | `active (running)` |
| **/usr/bin/python3 Service** | `systemctl status /usr/bin/python3` | `active (running)` |
| **nginx: Service** | `systemctl status nginx:` | `active (running)` |
| **nginx: Service** | `systemctl status nginx:` | `active (running)` |
| **/opt/learnchain/venv/bin/python Service** | `systemctl status /opt/learnchain/venv/bin/python` | `active (running)` |
| **/opt/learnchain/venv/bin/python Service** | `systemctl status /opt/learnchain/venv/bin/python` | `active (running)` |
| **Disk Space** | `df -h` | < 80% usage |
| **Memory** | `free -h` | Available memory > 1GB |
| **Processes** | `ps aux | grep python` | Application processes running |


## 🔧 Development

### Quick Links
- [🏗️ Developer Setup Guide](./developer_setup.md) - Complete setup instructions
- [📚 API Documentation](./api_documentation.md) - Full API reference  
- [🏢 Business Logic Guide](./business_logic.md) - Understanding the business
- [🚀 Deployment Guide](./deployment_runbook.md) - Production deployment
- [🔒 Security Assessment](./security_assessment.md) - Security considerations
- [🐛 Troubleshooting Guide](./troubleshooting.md) - Common issues and solutions

### Development Workflow
1. **Setup**: Follow the [Developer Setup Guide](./developer_setup.md)
2. **Code**: Make your changes following the established patterns
3. **Test**: Run the test suite (see Testing section below)
4. **Deploy**: Use the deployment runbook for production

### Testing
```bash
# Test API endpoints
curl -X GET http://localhost/health

```

## 🐛 Common Issues

- **Service Won't Start**: Check `systemctl status service-name` and logs in `/var/log/`
- **API Not Responding**: Verify process is running with `ps aux | grep python`
- **Permission Errors**: Check file ownership and permissions
- **Port Already in Use**: Find process with `netstat -tlnp | grep :port`
- **The script handles sensitive document data and uploads it to S3. Improper permissions or misconfigurations could lead to data exposure.**: Ensure S3 bucket policies and IAM roles are correctly configured to restrict access.

## 📊 Performance


| Metric | Current Status |
|--------|----------------|
| **Active Processes** | 15 processes |
| **Application Files** | 4 files |
| **Architecture Complexity** | Unknown |
| **Scalability Assessment** | Unknown |

### Performance Considerations
- The script processes one message at a time, which could become a bottleneck if the queue receives a high volume of messages.
- The use of a 15-minute visibility timeout suggests that parsing might take a significant amount of time, which could delay message processing.
- The script processes documents and extracts data in a sequential manner, which might be a bottleneck for large documents or high volumes of documents.


---

**🤖 Auto-generated by InfraDoc 2.0** - *"Developers just develop, we'll document"*  
*Last updated: 2025-06-23 22:02:16*

*This documentation was automatically generated by analyzing the actual running infrastructure. 
It reflects the real system architecture, APIs, and deployment patterns.*
