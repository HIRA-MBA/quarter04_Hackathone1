---
sidebar_position: 2
title: "Appendix B: Cloud Setup Guide"
description: Setting up cloud resources for simulation and training
---

# Appendix B: Cloud Setup Guide

This appendix covers cloud computing options for running GPU-intensive simulations and training when local hardware is insufficient.

## When to Use Cloud Resources

- Isaac Sim requires RTX GPU (not available locally)
- Training reinforcement learning models
- Running large-scale simulations
- Collaborative development environments

## AWS Setup

### EC2 Instance for Robotics

**Recommended Instance Types:**

| Instance | GPU | vCPUs | RAM | Use Case |
|----------|-----|-------|-----|----------|
| g4dn.xlarge | T4 | 4 | 16GB | Basic Isaac Sim |
| g5.2xlarge | A10G | 8 | 32GB | Full simulation |
| p3.2xlarge | V100 | 8 | 61GB | RL training |

**Launch Steps:**

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure

# Launch instance with Deep Learning AMI
aws ec2 run-instances \
  --image-id ami-0123456789abcdef0 \
  --instance-type g4dn.xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxx
```

### AWS RoboMaker

AWS RoboMaker provides managed ROS environments:

1. Navigate to AWS RoboMaker Console
2. Create Development Environment
3. Select ROS 2 Humble bundle
4. Connect via Cloud9 IDE

**Cost:** ~$0.50/hour for basic simulation

## Google Cloud Platform

### Compute Engine with GPU

```bash
# Create VM with NVIDIA GPU
gcloud compute instances create robotics-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE
```

### Install NVIDIA Drivers

```bash
# SSH into instance
gcloud compute ssh robotics-vm

# Install drivers
sudo apt update
sudo apt install -y nvidia-driver-535
sudo reboot
```

## Microsoft Azure

### Azure VM for Robotics

```bash
# Create resource group
az group create --name robotics-rg --location eastus

# Create VM with GPU
az vm create \
  --resource-group robotics-rg \
  --name robotics-vm \
  --image Ubuntu2204 \
  --size Standard_NC6s_v3 \
  --admin-username azureuser \
  --generate-ssh-keys
```

## Cost Optimization

### Spot/Preemptible Instances

Save 60-90% with interruptible instances:

| Provider | Service | Savings |
|----------|---------|---------|
| AWS | Spot Instances | Up to 90% |
| GCP | Preemptible VMs | Up to 80% |
| Azure | Spot VMs | Up to 90% |

**Best for:** Batch training, non-critical simulations

### Auto-Shutdown Scripts

```bash
#!/bin/bash
# auto-shutdown.sh - Stop instance after idle

IDLE_THRESHOLD=300  # 5 minutes

while true; do
  IDLE=$(xprintidle 2>/dev/null || echo 0)
  if [ "$IDLE" -gt "$((IDLE_THRESHOLD * 1000))" ]; then
    sudo shutdown -h now
  fi
  sleep 60
done
```

### Cost Estimates

| Workload | Hours/Week | AWS Cost | GCP Cost |
|----------|------------|----------|----------|
| Basic sim | 10 | ~$8 | ~$7 |
| Full sim | 20 | ~$40 | ~$35 |
| RL training | 40 | ~$150 | ~$130 |

## Remote Development Workflow

### VS Code Remote SSH

1. Install "Remote - SSH" extension
2. Add host to SSH config:

```
Host robotics-cloud
  HostName <instance-ip>
  User ubuntu
  IdentityFile ~/.ssh/your-key.pem
```

3. Connect via Command Palette: "Remote-SSH: Connect to Host"

### Port Forwarding for Visualization

```bash
# Forward Gazebo GUI
ssh -L 11345:localhost:11345 robotics-cloud

# Forward RViz
ssh -L 5900:localhost:5900 robotics-cloud

# Forward Isaac Sim streaming
ssh -L 8211:localhost:8211 robotics-cloud
```

### Syncing Code

```bash
# Using rsync
rsync -avz --exclude '.git' \
  ./workspace/ robotics-cloud:~/workspace/

# Using scp for single files
scp local_file.py robotics-cloud:~/workspace/
```

## Docker in Cloud

### Pull ROS 2 Images

```bash
# Official ROS 2 Humble
docker pull ros:humble

# NVIDIA Isaac ROS
docker pull nvcr.io/nvidia/isaac/ros:humble

# Run with GPU support
docker run --gpus all -it ros:humble
```

### Docker Compose for Full Stack

```yaml
version: '3.8'
services:
  ros2:
    image: ros:humble
    network_mode: host
    volumes:
      - ./workspace:/workspace
    command: ros2 launch my_robot sim.launch.py

  gazebo:
    image: gazebo:latest
    network_mode: host
    environment:
      - DISPLAY=:0
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix
```

## Security Best Practices

1. **SSH Keys Only** - Disable password authentication
2. **Security Groups** - Restrict ports to your IP
3. **VPC** - Isolate robotics instances
4. **IAM Roles** - Minimal permissions
5. **Encryption** - Enable EBS/disk encryption

## Quick Reference

### Start Development Session

```bash
# 1. Start cloud instance
aws ec2 start-instances --instance-ids i-xxxxx

# 2. Wait for running state
aws ec2 wait instance-running --instance-ids i-xxxxx

# 3. Get public IP
IP=$(aws ec2 describe-instances --instance-ids i-xxxxx \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

# 4. Connect
ssh -i key.pem ubuntu@$IP
```

### End Session (Save Costs!)

```bash
# Stop instance (preserves data)
aws ec2 stop-instances --instance-ids i-xxxxx

# Or terminate (deletes everything)
aws ec2 terminate-instances --instance-ids i-xxxxx
```
