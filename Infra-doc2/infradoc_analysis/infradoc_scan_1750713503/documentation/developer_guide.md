# Developer Infrastructure Recreation Guide

## Overview

This guide provides step-by-step instructions to recreate the analyzed infrastructure environment. Use this for development, testing, or disaster recovery scenarios.

**Target System**: ec2-3-143-6-83.us-east-2.compute.amazonaws.com  
**Architecture**: Microservices  
**Analysis Date**: 2025-06-23  

## Prerequisites

### System Requirements
- Linux-based operating system (Ubuntu 20.04+ recommended)
- Root or sudo access
- Internet connectivity for package installation

### Technology Stack Setup

Based on our analysis, install the following technologies:

- Python 3.x runtime and virtual environment
- Nginx web server

## Step-by-Step Recreation

### 1. System Preparation

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install essential development tools
sudo apt install -y build-essential curl wget git vim
```

### 2. Technology Installation

```bash
# Install Python and virtual environment
sudo apt install -y python3 python3-pip python3-venv

# Install Nginx
sudo apt install -y nginx
sudo systemctl enable nginx

```

### 3. Application Setup

#### Create Application Directories
```bash
sudo mkdir -p \opt\learnchain
```

#### Application Files Setup

Based on the discovered application files, create the following structure:

- `/opt/learnchain/create_course_knowledge_graph_neo.py` (Python)
- `/opt/learnchain/worker.py` (Python)
- `/opt/learnchain/parsing_adapter.py` (Python)
- `/opt/learnchain/worker-2.py` (Python)

### 4. Service Configuration

#### Process Management

The following services were identified and should be configured:


**nginx:** (background_worker)
- Purpose: Background task processing
- User: www-data
- Command: `nginx: worker process`


**/opt/learnchain/venv/bin/python** (background_worker)
- Purpose: Background task processing
- User: ubuntu
- Command: `/opt/learnchain/venv/bin/python /opt/learnchain/worker.py`


**/opt/learnchain/venv/bin/python** (background_worker)
- Purpose: Background task processing
- User: ubuntu
- Command: `/opt/learnchain/venv/bin/python /opt/learnchain/worker-2.py`


#### Systemd Service Configuration


Create systemd service file for nginx::

```bash
sudo tee /etc/systemd/system/nginx:.service > /dev/null <<EOF
[Unit]
Description=Background task processing
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=None
ExecStart=nginx: worker process
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable nginx:.service
sudo systemctl start nginx:.service
```


Create systemd service file for /opt/learnchain/venv/bin/python:

```bash
sudo tee /etc/systemd/system/optlearnchainvenvbinpython.service > /dev/null <<EOF
[Unit]
Description=Background task processing
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/learnchain
ExecStart=/opt/learnchain/venv/bin/python /opt/learnchain/worker.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable optlearnchainvenvbinpython.service
sudo systemctl start optlearnchainvenvbinpython.service
```


Create systemd service file for /opt/learnchain/venv/bin/python:

```bash
sudo tee /etc/systemd/system/optlearnchainvenvbinpython.service > /dev/null <<EOF
[Unit]
Description=Background task processing
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/learnchain
ExecStart=/opt/learnchain/venv/bin/python /opt/learnchain/worker-2.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable optlearnchainvenvbinpython.service
sudo systemctl start optlearnchainvenvbinpython.service
```


### 5. Configuration Files


Create necessary configuration files based on discovered patterns:

- /etc/systemd/system/*.service
- /etc/nginx/nginx.conf
- /etc/nginx/sites-available/*
- /opt/*/config/*
- /srv/*/config/*
- /var/www/*/config/*

**Note**: Review existing configuration files on the target system and adapt accordingly.


### 6. Networking and Security

#### Firewall Configuration
```bash
# Configure UFW firewall
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
```

#### SSL/TLS Setup (if web services detected)
```bash
# Install Certbot for Let's Encrypt
sudo apt install -y certbot python3-certbot-nginx

# Obtain SSL certificate (replace with your domain)
sudo certbot --nginx -d your-domain.com
```

### 7. Monitoring and Logging

#### Log Configuration
```bash
# Create log directories
sudo mkdir -p /var/log/app

# Configure log rotation
sudo tee /etc/logrotate.d/app > /dev/null <<EOF
/var/log/app/*.log {
    daily
    missingok
    rotate 30
    compress
    notifempty
    create 0644 ubuntu ubuntu
}
EOF
```

### 8. Validation and Testing

#### Service Health Checks
```bash
# Check service status
sudo systemctl status @dbus-daemon
sudo systemctl status usrsbinchronyd
sudo systemctl status usrsbinchronyd
sudo systemctl status usrlibpolkit-1polkitd
sudo systemctl status nginx
sudo systemctl status usrsbinrsyslogd
sudo systemctl status (sd-pam)
sudo systemctl status sshd:
sudo systemctl status -bash
sudo systemctl status sshd:
sudo systemctl status sshd:
sudo systemctl status sshd:
sudo systemctl status ps

# Check application processes
ps aux | grep python
ps aux | grep worker

# Check network connections
sudo netstat -tlnp
```

#### Application Testing
```bash
# Test web services
curl -I http://localhost

# Check application logs
tail -f /var/log/app/application.log

# Monitor system resources
htop
```

## Troubleshooting

### Common Issues

1. **Service Won't Start**
   - Check systemd service configuration
   - Verify file permissions
   - Check log files for errors

2. **Port Conflicts**
   - Use `netstat -tlnp` to check port usage
   - Modify service configurations as needed

3. **Permission Denied**
   - Verify user permissions on application directories
   - Check SELinux/AppArmor policies if applicable

### Log Locations
- Application logs: `/var/log/app/`
- System logs: `/var/log/syslog`
- Service logs: `journalctl -u service-name`

## Maintenance

### Regular Tasks
- Update system packages monthly
- Monitor disk space and log rotation
- Review security updates
- Backup application data and configurations

### Performance Monitoring
- Use `htop` for process monitoring
- Monitor disk usage with `df -h`
- Check memory usage with `free -h`

---

*Developer guide generated by InfraDoc 2.0*  
*Last updated: 2025-06-23 17:19:57*
