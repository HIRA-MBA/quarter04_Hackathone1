"""Email service for sending verification and password reset emails."""

import asyncio
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from app.config import get_settings

settings = get_settings()


class EmailService:
    """Service for sending transactional emails."""

    @classmethod
    def is_configured(cls) -> bool:
        """Check if email service is properly configured."""
        return bool(settings.smtp_user and settings.smtp_password)

    @classmethod
    async def send_email(
        cls,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: str | None = None,
    ) -> bool:
        """
        Send an email asynchronously.

        Args:
            to_email: Recipient email address
            subject: Email subject
            html_content: HTML body of the email
            text_content: Plain text alternative (optional)

        Returns:
            True if email was sent successfully, False otherwise
        """
        if not cls.is_configured():
            print(f"[Email Service] Not configured. Would send to {to_email}: {subject}")
            return False

        try:
            # Run SMTP operations in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                cls._send_email_sync,
                to_email,
                subject,
                html_content,
                text_content,
            )
            return True
        except Exception as e:
            print(f"[Email Service] Failed to send email: {e}")
            return False

    @classmethod
    def _send_email_sync(
        cls,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: str | None = None,
    ) -> None:
        """Synchronous email sending (runs in thread pool)."""
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = f"{settings.smtp_from_name} <{settings.smtp_from_email}>"
        message["To"] = to_email

        # Add plain text version
        if text_content:
            part1 = MIMEText(text_content, "plain")
            message.attach(part1)

        # Add HTML version
        part2 = MIMEText(html_content, "html")
        message.attach(part2)

        # Send email
        context = ssl.create_default_context()
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.starttls(context=context)
            server.login(settings.smtp_user, settings.smtp_password)
            server.sendmail(settings.smtp_from_email, to_email, message.as_string())

    @classmethod
    async def send_verification_email(
        cls, to_email: str, token: str, full_name: str | None = None
    ) -> bool:
        """Send email verification email."""
        verification_url = f"{settings.frontend_url}/auth/verify-email?token={token}"
        name = full_name or "there"

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Verify Your Email</title>
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px 10px 0 0;">
        <h1 style="color: white; margin: 0; font-size: 24px;">Physical AI Textbook</h1>
    </div>

    <div style="background: #ffffff; padding: 30px; border: 1px solid #e0e0e0; border-top: none; border-radius: 0 0 10px 10px;">
        <h2 style="color: #333; margin-top: 0;">Verify Your Email Address</h2>

        <p>Hi {name},</p>

        <p>Thanks for signing up for the Physical AI & Humanoid Robotics Textbook! Please verify your email address by clicking the button below:</p>

        <div style="text-align: center; margin: 30px 0;">
            <a href="{verification_url}" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 14px 30px; text-decoration: none; border-radius: 5px; font-weight: bold; display: inline-block;">Verify Email Address</a>
        </div>

        <p style="color: #666; font-size: 14px;">Or copy and paste this link into your browser:</p>
        <p style="color: #667eea; font-size: 14px; word-break: break-all;">{verification_url}</p>

        <p style="color: #666; font-size: 14px;">This link will expire in 24 hours.</p>

        <hr style="border: none; border-top: 1px solid #e0e0e0; margin: 30px 0;">

        <p style="color: #999; font-size: 12px; margin-bottom: 0;">
            If you didn't create an account with Physical AI Textbook, you can safely ignore this email.
        </p>
    </div>
</body>
</html>
"""

        text_content = f"""
Verify Your Email Address

Hi {name},

Thanks for signing up for the Physical AI & Humanoid Robotics Textbook! Please verify your email address by clicking the link below:

{verification_url}

This link will expire in 24 hours.

If you didn't create an account with Physical AI Textbook, you can safely ignore this email.
"""

        return await cls.send_email(
            to_email=to_email,
            subject="Verify your email - Physical AI Textbook",
            html_content=html_content,
            text_content=text_content,
        )

    @classmethod
    async def send_password_reset_email(
        cls, to_email: str, token: str, full_name: str | None = None
    ) -> bool:
        """Send password reset email."""
        reset_url = f"{settings.frontend_url}/auth/reset-password?token={token}"
        name = full_name or "there"

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reset Your Password</title>
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px 10px 0 0;">
        <h1 style="color: white; margin: 0; font-size: 24px;">Physical AI Textbook</h1>
    </div>

    <div style="background: #ffffff; padding: 30px; border: 1px solid #e0e0e0; border-top: none; border-radius: 0 0 10px 10px;">
        <h2 style="color: #333; margin-top: 0;">Reset Your Password</h2>

        <p>Hi {name},</p>

        <p>We received a request to reset your password. Click the button below to create a new password:</p>

        <div style="text-align: center; margin: 30px 0;">
            <a href="{reset_url}" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 14px 30px; text-decoration: none; border-radius: 5px; font-weight: bold; display: inline-block;">Reset Password</a>
        </div>

        <p style="color: #666; font-size: 14px;">Or copy and paste this link into your browser:</p>
        <p style="color: #667eea; font-size: 14px; word-break: break-all;">{reset_url}</p>

        <p style="color: #666; font-size: 14px;">This link will expire in 1 hour.</p>

        <hr style="border: none; border-top: 1px solid #e0e0e0; margin: 30px 0;">

        <p style="color: #999; font-size: 12px; margin-bottom: 0;">
            If you didn't request a password reset, you can safely ignore this email. Your password will remain unchanged.
        </p>
    </div>
</body>
</html>
"""

        text_content = f"""
Reset Your Password

Hi {name},

We received a request to reset your password. Click the link below to create a new password:

{reset_url}

This link will expire in 1 hour.

If you didn't request a password reset, you can safely ignore this email. Your password will remain unchanged.
"""

        return await cls.send_email(
            to_email=to_email,
            subject="Reset your password - Physical AI Textbook",
            html_content=html_content,
            text_content=text_content,
        )

    @classmethod
    async def send_welcome_email(cls, to_email: str, full_name: str | None = None) -> bool:
        """Send welcome email after successful verification."""
        name = full_name or "there"
        dashboard_url = f"{settings.frontend_url}/dashboard"

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome to Physical AI Textbook</title>
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px 10px 0 0;">
        <h1 style="color: white; margin: 0; font-size: 24px;">Physical AI Textbook</h1>
    </div>

    <div style="background: #ffffff; padding: 30px; border: 1px solid #e0e0e0; border-top: none; border-radius: 0 0 10px 10px;">
        <h2 style="color: #333; margin-top: 0;">Welcome to Physical AI & Humanoid Robotics!</h2>

        <p>Hi {name},</p>

        <p>Your email has been verified and your account is now active. You're ready to start learning about:</p>

        <ul style="color: #555;">
            <li><strong>ROS 2 Fundamentals</strong> - Build your first robot nodes</li>
            <li><strong>Digital Twins</strong> - Simulate with Gazebo and Unity</li>
            <li><strong>NVIDIA Isaac</strong> - GPU-accelerated robotics</li>
            <li><strong>Vision-Language-Action Models</strong> - Cutting-edge AI for robots</li>
        </ul>

        <div style="text-align: center; margin: 30px 0;">
            <a href="{dashboard_url}" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 14px 30px; text-decoration: none; border-radius: 5px; font-weight: bold; display: inline-block;">Start Learning</a>
        </div>

        <hr style="border: none; border-top: 1px solid #e0e0e0; margin: 30px 0;">

        <p style="color: #999; font-size: 12px; margin-bottom: 0;">
            Happy learning!<br>
            The Physical AI Textbook Team
        </p>
    </div>
</body>
</html>
"""

        text_content = f"""
Welcome to Physical AI & Humanoid Robotics!

Hi {name},

Your email has been verified and your account is now active. You're ready to start learning about:

- ROS 2 Fundamentals - Build your first robot nodes
- Digital Twins - Simulate with Gazebo and Unity
- NVIDIA Isaac - GPU-accelerated robotics
- Vision-Language-Action Models - Cutting-edge AI for robots

Start learning: {dashboard_url}

Happy learning!
The Physical AI Textbook Team
"""

        return await cls.send_email(
            to_email=to_email,
            subject="Welcome to Physical AI Textbook!",
            html_content=html_content,
            text_content=text_content,
        )
