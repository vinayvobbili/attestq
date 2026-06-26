"""A self-contained sample assessment.

Everything here is fictional — no real vendor, person, or document. The evidence
is written so most controls are clearly Met while a few (secure SDLC, log
retention, fourth-party management) are deliberately unaddressed, so a first run
shows the confidence gate returning "insufficient evidence" exactly as it behaves
on real data.

Used by ``attestq demo`` and the example scripts. Import and reuse freely.
"""

from __future__ import annotations

from typing import Dict, List

from .models import Question, Questionnaire

DEMO_NAMESPACE = "helios"

# --- evidence corpus ----------------------------------------------------------

DEMO_DOCUMENTS: List[Dict] = [
    {
        "source": "Helios_SOC2_TypeII_2025.txt",
        "text": (
            "HELIOS DATA SYSTEMS - SOC 2 TYPE II REPORT. Trust Services Criteria in "
            "scope: Security, Availability, and Confidentiality. Coverage period: 12 "
            "months. The subservice cloud infrastructure provider is presented using "
            "the carve-out method. Controls tested covered logical access, encryption, "
            "change management, incident response, and vulnerability management. Two "
            "minor exceptions were noted around timely access removal and a single "
            "missing code-review record; both have management remediation responses."
        ),
    },
    {
        "source": "Helios_Information_Security_Policy.txt",
        "text": (
            "HELIOS DATA SYSTEMS - INFORMATION SECURITY POLICY. Helios maintains a "
            "formal information security program aligned to ISO 27001 and the NIST "
            "Cybersecurity Framework, owned by the CISO and approved by executive "
            "leadership at least annually. A formal risk assessment is performed "
            "annually and upon significant change, tracked in a risk register. All "
            "personnel complete security awareness training at hire and annually; "
            "phishing simulations run quarterly. Background checks are performed on "
            "new hires where permitted by law."
        ),
    },
    {
        "source": "Helios_Access_Control_Standard.txt",
        "text": (
            "HELIOS DATA SYSTEMS - ACCESS CONTROL & IDENTITY STANDARD. Multi-factor "
            "authentication is enforced for all remote access and for all privileged "
            "and administrative accounts without exception. Access is granted on a "
            "least-privilege, role-based basis and reviewed quarterly by system "
            "owners. Access is provisioned and de-provisioned through a ticketed "
            "workflow; shared administrative credentials are prohibited and all "
            "privileged sessions are logged."
        ),
    },
    {
        "source": "Helios_Data_Protection_Standard.txt",
        "text": (
            "HELIOS DATA SYSTEMS - DATA PROTECTION & ENCRYPTION STANDARD. All data in "
            "transit is encrypted using TLS 1.2 or higher; TLS 1.0 and 1.1 are "
            "disabled. All customer data at rest is encrypted using AES-256 at the "
            "storage and database layers. Cryptographic keys are managed in a "
            "dedicated key management service, access-restricted and rotated at least "
            "annually. Data is classified and securely destroyed at end of life."
        ),
    },
    {
        "source": "Helios_Network_and_VulnMgmt_Overview.txt",
        "text": (
            "HELIOS DATA SYSTEMS - NETWORK SECURITY & VULNERABILITY MANAGEMENT. "
            "Production is segmented from corporate networks; stateful firewalls and "
            "an IDS/IPS monitor production traffic, and administrative access passes "
            "through a bastion host with MFA. An EDR agent is deployed to all servers "
            "and workstations. Authenticated vulnerability scans run weekly with "
            "remediation SLAs of 14 days for critical and 30 days for high findings. "
            "An independent third party performs penetration testing at least annually."
        ),
    },
    {
        "source": "Helios_Business_Continuity_Summary.txt",
        "text": (
            "HELIOS DATA SYSTEMS - BUSINESS CONTINUITY & DISASTER RECOVERY. Documented "
            "BCP and DR plans are reviewed and tested at least annually via tabletop "
            "and failover exercises. Recovery objectives for the production platform "
            "are an RTO of 4 hours and an RPO of 1 hour. Backups are performed daily, "
            "encrypted, and replicated to a geographically separate region. A "
            "documented incident response plan defines roles, severity levels, and "
            "escalation paths, and customers are notified of confirmed incidents "
            "without undue delay."
        ),
    },
]

# --- questionnaire ------------------------------------------------------------

_CHOICES = ["Met", "Not Met", "Not Applicable"]

DEMO_QUESTIONS = [
    ("GRC-1", "Governance", "Does the vendor maintain a formal information security program with executive ownership and annual review?"),
    ("GRC-2", "Governance", "Does the vendor perform periodic risk assessments and track findings to remediation?"),
    ("IAM-1", "Identity & Access", "Is multi-factor authentication enforced for remote and privileged access?"),
    ("IAM-2", "Identity & Access", "Are access rights granted on least privilege and reviewed periodically?"),
    ("DATA-1", "Data Protection", "Is customer data encrypted in transit and at rest using industry-standard algorithms?"),
    ("NET-1", "Network & Vulnerability", "Does the vendor run regular vulnerability scanning with defined remediation SLAs?"),
    ("RES-1", "Resilience", "Does the vendor maintain and test business continuity and disaster recovery plans?"),
    ("IR-1", "Incident Response", "Does the vendor have a documented incident response and customer-notification process?"),
    # --- deliberate evidence gaps (expect "insufficient evidence") ---
    ("SDLC-1", "Secure Development", "Does the vendor follow a secure software development lifecycle with SAST/DAST and mandatory code review?"),
    ("LOG-1", "Logging & Monitoring", "Does the vendor centralize security event logs and define explicit log-retention periods?"),
    ("TPC-1", "Fourth-Party Risk", "Does the vendor assess and manage the security of its own subcontractors and fourth parties?"),
]


def demo_questionnaire() -> Questionnaire:
    """The bundled sample Questionnaire."""
    return Questionnaire(
        id="vendor-security-review",
        title="Vendor Security Due-Diligence Review (Sample)",
        questions=[
            Question(id=qid, prompt=prompt, choices=_CHOICES, domain=domain)
            for qid, domain, prompt in DEMO_QUESTIONS
        ],
    )
