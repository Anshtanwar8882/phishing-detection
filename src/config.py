"""Single source of truth: states, observations, generator parameters, split settings."""
import numpy as np

SEED = 42

# ---- Hidden campaign stages (K = 6) -------------------------------------------------
STATES = [
    "Benign",              # 0 background / campaign inactive
    "Reconnaissance",      # 1 attacker prepares (domain probes, sender probes)
    "Delivery",            # 2 phishing email arrives
    "Lure/Click",          # 3 victim engages (click, landing page)
    "Credential Harvest",  # 4 credentials submitted
    "Takeover",            # 5 account abuse / exfiltration
]
K = len(STATES)
HARVEST_STAGES = [4, 5]            # alert target: Credential Harvest or Takeover

# ---- Observed events (M = 12) ---------------------------------------------------------
OBS = [
    "normal_activity",        # 0
    "email_clean",            # 1  real email: no URL, no urgency keywords
    "email_urgent_kw",        # 2  real email: urgency/credential keywords, no URL
    "email_url",              # 3  real email: contains URL
    "email_url_urgent",       # 4  real email: URL + urgency keywords
    "domain_probe",           # 5  synthetic
    "link_clicked",           # 6  synthetic
    "landing_page_visit",     # 7  synthetic
    "credential_submitted",   # 8  synthetic
    "new_ip_login",           # 9  synthetic
    "mailbox_rule_created",   # 10 synthetic
    "bulk_forward",           # 11 synthetic
]
M = len(OBS)
EMAIL_OBS = [1, 2, 3, 4]
OBS_ID = {n: i for i, n in enumerate(OBS)}
RULE_OBS = [OBS_ID["credential_submitted"], OBS_ID["mailbox_rule_created"], OBS_ID["bulk_forward"]]

# ---- Real-data window (CEAS-08: 99%+ of emails fall in this week) ----------------------
WINDOW_START = "2008-08-04"
WINDOW_END = "2008-08-11"
N_PER_CLASS = 3000                  # real emails sampled per class (seeds of episodes)
SPLIT_FRACS = (0.70, 0.15, 0.15)    # chronological train / val / test
SEQ_LEN_RANGE = (8, 16)             # inclusive

# ---- Documented synthetic generator ---------------------------------------------------
# Transition matrix of a campaign episode. Row 0 (Benign) is absorbing: a campaign that
# "fizzles" (victim ignores it) returns to background and stays there.
A_GEN = np.array([
    #  Ben   Rec   Del   Lure  Harv  Take
    [1.00, 0.00, 0.00, 0.00, 0.00, 0.00],  # Benign
    [0.05, 0.55, 0.38, 0.02, 0.00, 0.00],  # Reconnaissance
    [0.22, 0.00, 0.35, 0.40, 0.03, 0.00],  # Delivery
    [0.20, 0.00, 0.00, 0.40, 0.38, 0.02],  # Lure/Click
    [0.10, 0.00, 0.00, 0.00, 0.40, 0.50],  # Credential Harvest
    [0.08, 0.00, 0.00, 0.00, 0.00, 0.92],  # Takeover
])
# Same chain but with far more stage skipping (used only for the robustness test)
A_SKIP = np.array([
    [1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.05, 0.40, 0.25, 0.20, 0.10, 0.00],
    [0.15, 0.00, 0.25, 0.25, 0.30, 0.05],
    [0.15, 0.00, 0.00, 0.30, 0.35, 0.20],
    [0.10, 0.00, 0.00, 0.00, 0.30, 0.60],
    [0.08, 0.00, 0.00, 0.00, 0.00, 0.92],
])
PHISH_START = np.array([0.0, 0.35, 0.65, 0.0, 0.0, 0.0])   # phishing episodes start here
BENIGN_START = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Emission weights per state (normalised in code). "EMAIL_PH"/"EMAIL_BN" are spread over the
# four email categories using the REAL phishing / benign email category mix of the split.
EMISSION_WEIGHTS = {
    0: {"normal_activity": .634, "EMAIL_BN": .24, "domain_probe": .005, "link_clicked": .05,
        "landing_page_visit": .01, "credential_submitted": .004, "new_ip_login": .05,
        "mailbox_rule_created": .004, "bulk_forward": .003},
    1: {"normal_activity": .55, "EMAIL_BN": .05, "domain_probe": .30, "link_clicked": .01,
        "new_ip_login": .03, "landing_page_visit": .005, "bulk_forward": .005},
    2: {"normal_activity": .18, "EMAIL_PH": .72, "domain_probe": .02, "link_clicked": .05,
        "landing_page_visit": .01, "new_ip_login": .01, "credential_submitted": .005, "bulk_forward": .005},
    3: {"normal_activity": .20, "EMAIL_PH": .10, "link_clicked": .35, "landing_page_visit": .30,
        "credential_submitted": .02, "new_ip_login": .01, "domain_probe": .01, "bulk_forward": .01},
    4: {"normal_activity": .20, "EMAIL_PH": .03, "link_clicked": .08, "landing_page_visit": .20,
        "credential_submitted": .42, "new_ip_login": .04, "mailbox_rule_created": .02, "bulk_forward": .01},
    5: {"normal_activity": .22, "EMAIL_PH": .02, "new_ip_login": .30, "mailbox_rule_created": .24,
        "bulk_forward": .18, "credential_submitted": .02, "link_clicked": .02},
}
# mean gap (minutes) between consecutive events, by the state of the later event
GAP_MEAN_MIN = {0: 240, 1: 300, 2: 30, 3: 25, 4: 20, 5: 90}
