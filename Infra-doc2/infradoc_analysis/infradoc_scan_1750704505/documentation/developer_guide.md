# Developer Infrastructure Recreation Guide

## Overview

This guide provides step-by-step instructions to recreate the analyzed infrastructure environment. Use this for development, testing, or disaster recovery scenarios.

**Target System**: ec2-3-143-6-83.us-east-2.compute.amazonaws.com  
**Architecture**: Standard deployment  
**Analysis Date**: 2025-06-23  

## Prerequisites

### System Requirements
- Linux-based operating system (Ubuntu 20.04+ recommended)
- Root or sudo access
- Internet connectivity for package installation

### Technology Stack Setup

Based on our analysis, install the following technologies:

- Basic Linux utilities

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
```

### 3. Application Setup

#### Create Application Directories
```bash
sudo mkdir -p \opt\learnchain
```

#### Application Files Setup

Based on the discovered application files, create the following structure:

- `/opt/learnchain/create_course_knowledge_graph_neo.py` (Python)
- `/opt/learnchain/worker-2.py` (Python)

### 4. Service Configuration

#### Process Management

The following services were identified and should be configured:


**[kworker/R-rcu_g]** (unknown)
- Purpose: Basic process discovery
- User: root
- Command: `[kworker/R-rcu_g]`


**[kworker/R-rcu_p]** (unknown)
- Purpose: Basic process discovery
- User: root
- Command: `[kworker/R-rcu_p]`


**[kworker/R-slub_]** (unknown)
- Purpose: Basic process discovery
- User: root
- Command: `[kworker/R-slub_]`


**[kworker/R-netns]** (unknown)
- Purpose: Basic process discovery
- User: root
- Command: `[kworker/R-netns]`


**[kworker/0:0H-events_highpri]** (unknown)
- Purpose: Basic process discovery
- User: root
- Command: `[kworker/0:0H-events_highpri]`


**[kworker/R-mm_pe]** (unknown)
- Purpose: Basic process discovery
- User: root
- Command: `[kworker/R-mm_pe]`


**[kworker/R-inet_]** (unknown)
- Purpose: Basic process discovery
- User: root
- Command: `[kworker/R-inet_]`


#### Systemd Service Configuration


Create systemd service file for [kworker/R-rcu_g]:

```bash
sudo tee /etc/systemd/system/[kworkerR-rcu_g].service > /dev/null <<EOF
[Unit]
Description=Basic process discovery
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=None
ExecStart=[kworker/R-rcu_g]
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable [kworkerR-rcu_g].service
sudo systemctl start [kworkerR-rcu_g].service
```


Create systemd service file for [kworker/R-rcu_p]:

```bash
sudo tee /etc/systemd/system/[kworkerR-rcu_p].service > /dev/null <<EOF
[Unit]
Description=Basic process discovery
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=None
ExecStart=[kworker/R-rcu_p]
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable [kworkerR-rcu_p].service
sudo systemctl start [kworkerR-rcu_p].service
```


Create systemd service file for [kworker/R-slub_]:

```bash
sudo tee /etc/systemd/system/[kworkerR-slub_].service > /dev/null <<EOF
[Unit]
Description=Basic process discovery
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=None
ExecStart=[kworker/R-slub_]
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable [kworkerR-slub_].service
sudo systemctl start [kworkerR-slub_].service
```


Create systemd service file for [kworker/R-netns]:

```bash
sudo tee /etc/systemd/system/[kworkerR-netns].service > /dev/null <<EOF
[Unit]
Description=Basic process discovery
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=None
ExecStart=[kworker/R-netns]
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable [kworkerR-netns].service
sudo systemctl start [kworkerR-netns].service
```


Create systemd service file for [kworker/0:0H-events_highpri]:

```bash
sudo tee /etc/systemd/system/[kworker0:0H-events_highpri].service > /dev/null <<EOF
[Unit]
Description=Basic process discovery
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=None
ExecStart=[kworker/0:0H-events_highpri]
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable [kworker0:0H-events_highpri].service
sudo systemctl start [kworker0:0H-events_highpri].service
```


Create systemd service file for [kworker/R-mm_pe]:

```bash
sudo tee /etc/systemd/system/[kworkerR-mm_pe].service > /dev/null <<EOF
[Unit]
Description=Basic process discovery
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=None
ExecStart=[kworker/R-mm_pe]
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable [kworkerR-mm_pe].service
sudo systemctl start [kworkerR-mm_pe].service
```


Create systemd service file for [kworker/R-inet_]:

```bash
sudo tee /etc/systemd/system/[kworkerR-inet_].service > /dev/null <<EOF
[Unit]
Description=Basic process discovery
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=None
ExecStart=[kworker/R-inet_]
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable [kworkerR-inet_].service
sudo systemctl start [kworkerR-inet_].service
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
# No additional ports detected
```

#### SSL/TLS Setup (if web services detected)
No web services detected requiring SSL configuration.

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

# Check application processes
ps aux | grep python
ps aux | grep worker

# Check network connections
sudo netstat -tlnp
```

#### Application Testing
```bash
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
*Last updated: 2025-06-23 14:49:01*
