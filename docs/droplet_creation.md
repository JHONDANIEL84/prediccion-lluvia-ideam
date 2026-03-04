# Step-by-Step Guide: Create a DigitalOcean Droplet

Follow these steps to create your server (Droplet) in DigitalOcean.

## 1. Create a New Droplet
- Log in to your [DigitalOcean Control Panel](https://cloud.digitalocean.com/).
- Click the **Create** button at the top right and select **Droplets**.

## 2. Choose Region
- Select a region closest to you or your users (e.g., **New York** or **San Francisco**).

## 3. Choose an Image (Operating System)
- Select **Ubuntu 24.04 (LTS)**. This is a very stable and recommended OS for Docker.

## 4. Choose Size
- Choose the **Basic** plan.
- Under "CPU options", select **Regular**.
- Select the **$4/mo or $6/mo** plan (1 GB CPU, 512MB-1GB RAM, 10-25GB SSD). This is more than enough for a Streamlit app.

## 5. Choose Authentication
- **SSH Key (Recommended):** If you have an SSH key, add it here for better security.
- **Password:** If you prefer a password, create a strong one and **keep it safe**. You will need this to log in via terminal.

## 6. Finalize and Create
- Leave the other options (Backups, Monitoring) as default for now.
- Click the **Create Droplet** button at the bottom.

## 7. Get the IP Address
- Once the Droplet is created, you will see a public **IPv4 address** (e.g., `157.230.1.2`).

