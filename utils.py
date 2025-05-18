import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from config.config import config

import emails  # type: ignore
from jinja2 import Template

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmailData:
    html_content: str
    subject: str


def render_email_template(*, template_name: str, context: dict[str, Any]) -> str:
    template_str = (
        Path(__file__).parent / "email-templates" / "build" / template_name
    ).read_text()
    html_content = Template(template_str).render(context)
    return html_content


def send_email(
    *,
    email_to: str,
    subject: str = "",
    html_content: str = "",
) -> None:
    message = emails.Message(
        subject=subject,
        html=html_content,
        mail_from=(config.email.EMAILS_FROM_NAME, config.email.EMAILS_FROM_EMAIL),
    )
    smtp_options = {"host": config.email.SMTP_HOST, "port": config.email.SMTP_PORT}
    if config.email.SMTP_TLS:
        smtp_options["tls"] = True
    elif config.email.SMTP_SSL:
        smtp_options["ssl"] = True
    if config.email.SMTP_USER:
        smtp_options["user"] = config.email.SMTP_USER
    if config.email.SMTP_PASSWORD:
        smtp_options["password"] = config.email.SMTP_PASSWORD
    response = message.send(to=email_to, smtp=smtp_options)
    logger.info(f"send email result: {response}")


def generate_test_email(email_to: str) -> EmailData:
    project_name = config.email.PROJECT_NAME
    subject = f"{project_name} - Test email"
    html_content = render_email_template(
        template_name="test_email.html",
        context={"project_name": project_name, "email": email_to},
    )
    return EmailData(html_content=html_content, subject=subject)


def generate_fallback_email(user_query: str, chatbot_response: str, confidence_score: float) -> EmailData:
    project_name = config.email.PROJECT_NAME
    subject = f"{project_name} - Support Request"
    html_content = render_email_template(
        template_name="fallback_support.html",
        context={
            "project_name": project_name,
            "user_query": user_query,
            "chatbot_response": chatbot_response,
            "confidence_score": confidence_score
        },
    )
    return EmailData(html_content=html_content, subject=subject)