import csv
import random

benign_domains = [
    "google.com", "youtube.com", "facebook.com", "wikipedia.org", "twitter.com",
    "amazon.com", "reddit.com", "linkedin.com", "microsoft.com", "netflix.com",
    "instagram.com", "apple.com", "github.com", "stackoverflow.com", "medium.com",
    "yahoo.com", "bing.com", "cnn.com", "nytimes.com", "bbc.co.uk",
    "cloudflare.com", "spotify.com", "adobe.com", "dropbox.com", "quora.com",
    "twitch.tv", "salesforce.com", "zoom.us", "mozilla.org", "ebay.com",
    "walmart.com", "nih.gov", "stanford.edu", "mit.edu", "harvard.edu",
    "python.org", "pypi.org", "oracle.com", "ibm.com", "intel.com"
]

benign_paths = [
    "", "/about", "/contact", "/terms", "/privacy", "/help", "/documentation",
    "/products/features", "/blog/post/2026/04/update", "/search?q=technology",
    "/category/news/article-1029", "/explore/popular", "/download/latest",
    "/profile/user?id=10283", "/support/knowledge-base/faq"
]

phishing_patterns = [
    "paypal-security-update-{num}.com/signin",
    "secure-login-{name}.xyz/account-verification",
    "appleid-verify-alert-{num}.info/auth",
    "netflix-billing-update-{name}.com/payment",
    "wellsfargo-online-banking-{num}.net/login.php",
    "chase-security-check-{name}.top/portal",
    "bankofamerica-customer-support-{num}.site/verify",
    "google-docs-secure-view-{num}.biz/login",
    "microsoft-365-pass-recovery-{name}.club/auth",
    "amazon-order-cancelled-{num}.com/re-authenticate",
    "facebook-security-badge-{num}.me/appeal",
    "instagram-copyright-infringement-{num}.online/verify",
    "binance-kyc-update-{name}.cc/wallet",
    "metamask-seed-phrase-restore-{num}.io/connect",
    "coinbase-security-lockout-{name}.xyz/unlock"
]

malware_patterns = [
    "http://{ip}/trojan_{num}.exe",
    "http://{ip}:8080/payload_{name}.bin",
    "http://download-free-cracks-{num}.ru/setup.exe",
    "http://system-speedup-utility-{name}.cc/patch.vbs",
    "http://win-driver-update-{num}.top/driver_pack.scr",
    "http://flash-player-update-{name}.club/install_flash.exe",
    "http://critical-system-security-patch-{num}.biz/agent.dll",
    "http://{ip}/worm_spread_{name}.ps1",
    "http://free-crypto-miner-{num}.online/miner_svc.exe",
    "http://invoice-attached-payment-{num}.ws/invoice_1092.exe",
    "http://{ip}:4444/meterpreter_session_{num}.elf",
    "http://discord-nitro-gift-generator-{num}.ru/generator.exe",
    "http://ransom-decrypt-tool-{name}.cc/decryptor.bat",
    "http://direct-download-games-{num}.su/crack_patch.iso"
]

defacement_patterns = [
    "http://{domain}/deface_{name}.html",
    "http://{domain}/index.php?hacked_by=dark_ghost_{num}",
    "http://{domain}/warning_defaced_by_team_{name}.txt",
    "http://{domain}/defacement_mirror.php?id={num}",
    "http://{domain}/hacked.html",
    "http://{domain}/cyber_caliphate_ownz_{num}.htm",
    "http://{domain}/owned_by_{name}_team.html",
    "http://{domain}/index_defaced.asp",
    "http://{domain}/defaced-by-ghost-squad-{num}.html",
    "http://{domain}/pwned_by_{name}.php"
]

names = ["alpha", "delta", "shadow", "cyber", "phantom", "root", "echo", "knight", "phoenix", "anon"]
defaced_targets = [
    "local-council-services.org", "state-school-portal.edu", "city-water-dept.gov",
    "regional-chamber-commerce.com", "community-clinic-care.org", "smallbiz-catering.co",
    "library-network.org", "heritage-museum.org", "county-sheriff-office.org"
]

records = []

# Class 0: Benign
for domain in benign_domains:
    proto = "https://"
    for path in benign_paths:
        records.append((f"{proto}{domain}{path}", 0))

# Class 1: Phishing
for pattern in phishing_patterns:
    for i in range(35):
        name = random.choice(names)
        num = random.randint(100, 9999)
        url = "https://" + pattern.format(num=num, name=name)
        records.append((url, 1))

# Class 2: Malware
for pattern in malware_patterns:
    for i in range(35):
        ip = f"{random.randint(45, 218)}.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}"
        num = random.randint(100, 9999)
        name = random.choice(names)
        url = pattern.format(ip=ip, num=num, name=name)
        records.append((url, 2))

# Class 3: Defacement
for pattern in defacement_patterns:
    for domain in defaced_targets:
        for i in range(5):
            name = random.choice(names)
            num = random.randint(10, 999)
            url = pattern.format(domain=domain, name=name, num=num)
            records.append((url, 3))

random.seed(42)
random.shuffle(records)

with open("malicious_urls.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["url", "label"])
    for url, label in records:
        writer.writerow([url, label])

print(f"Generated {len(records)} balanced dataset rows in malicious_urls.csv")
