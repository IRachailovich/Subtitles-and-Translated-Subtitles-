import base64
import hmac
import ipaddress
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import time
from http.cookies import SimpleCookie
from pathlib import Path


ACCESS_COOKIE_NAME = "SubGen-Access"
TUNNEL_INTERFACE_PATTERN = re.compile(
    r"(?:vpn|tun|tap|wireguard|openvpn|tailscale|zerotier|hamachi|hyper-v|vmware|virtualbox|docker|wsl)",
    re.IGNORECASE,
)


def detect_client_platform(user_agent):
    value = (user_agent or "").lower()
    if any(marker in value for marker in ("iphone", "ipad", "ipod")):
        return "ios"
    # iPadOS can request a desktop-class user agent while retaining Mobile/.
    if "macintosh" in value and "mobile/" in value:
        return "ios"
    if "android" in value:
        return "android"
    return "other"


def detect_client_browser(user_agent):
    value = (user_agent or "").lower()
    if "samsungbrowser/" in value:
        return "samsung"
    if "crios/" in value:
        return "chrome"
    if "fxios/" in value:
        return "firefox"
    if "edgios/" in value:
        return "edge"
    if "opr/" in value or "opios/" in value:
        return "opera"
    if "edga/" in value or "edg/" in value:
        return "edge"
    if "firefox/" in value:
        return "firefox"
    if "chrome/" in value:
        return "chrome"
    if "safari/" in value and "version/" in value:
        return "safari"
    return "other"


def generate_access_token():
    return secrets.token_urlsafe(32)


def load_or_create_access_token(path):
    token_path = Path(path)
    try:
        token = token_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        token = ""
    if len(token) < 32:
        token = generate_access_token()
        write_access_token(token_path, token)
    return token


def write_access_token(path, token):
    token_path = Path(path)
    token_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = token_path.with_suffix(token_path.suffix + ".tmp")
    temporary_path.write_text(token + "\n", encoding="utf-8")
    try:
        os.chmod(temporary_path, 0o600)
    except OSError:
        pass
    temporary_path.replace(token_path)


def rotate_access_token(path):
    token = generate_access_token()
    write_access_token(path, token)
    return token


def is_loopback_address(address):
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return False


def token_from_cookie(cookie_header):
    if not cookie_header:
        return None
    cookie = SimpleCookie()
    try:
        cookie.load(cookie_header)
    except Exception:
        return None
    morsel = cookie.get(ACCESS_COOKIE_NAME)
    return morsel.value if morsel else None


def request_token(headers):
    authorization = headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    explicit = headers.get("X-SubGen-Token")
    if explicit:
        return explicit.strip()
    return token_from_cookie(headers.get("Cookie"))


def token_matches(candidate, expected):
    if not candidate or not expected:
        return False
    return hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


def interface_is_tunnel(name, description=""):
    return bool(TUNNEL_INTERFACE_PATTERN.search(f"{name} {description}"))


def interface_kind(name, description=""):
    value = f"{name} {description}".lower()
    if any(marker in value for marker in ("wi-fi", "wifi", "wireless", "wlan")):
        return "Wi-Fi"
    wired_interface = re.search(r"(?:^|\s)(?:eth\d*|en\d+)(?:\s|$)", value)
    if any(marker in value for marker in ("ethernet", "local area")) or wired_interface:
        return "Ethernet"
    return "Network"


def _discover_windows_ipv4_candidates():
    if os.name != "nt":
        return []
    script = r"""
$rows = @(Get-NetIPAddress -AddressFamily IPv4 -AddressState Preferred -ErrorAction SilentlyContinue | ForEach-Object {
    $address = $_
    $adapter = Get-NetAdapter -InterfaceIndex $address.InterfaceIndex -ErrorAction SilentlyContinue
    if ($adapter -and $adapter.Status -eq 'Up') {
        [pscustomobject]@{
            address = $address.IPAddress
            interface = $address.InterfaceAlias
            description = $adapter.InterfaceDescription
        }
    }
})
$rows | ConvertTo-Json -Depth 3 -Compress
"""
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            capture_output=True,
            text=True,
            timeout=8,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []
        rows = json.loads(result.stdout.strip())
        return rows if isinstance(rows, list) else [rows]
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []


def discover_private_ipv4_candidates():
    candidates = []
    for entry in _discover_windows_ipv4_candidates():
        interface = entry.get("interface", "")
        description = entry.get("description", "")
        address = entry.get("address", "")
        if interface_is_tunnel(interface, description) or not is_usable_private_ipv4(address):
            continue
        candidates.append({
            "address": address,
            "interface": interface,
            "kind": interface_kind(interface, description),
            "is_tunnel": False,
        })

    try:
        import psutil

        stats = psutil.net_if_stats()
        for interface, addresses in psutil.net_if_addrs().items():
            if interface in stats and not stats[interface].isup:
                continue
            if interface_is_tunnel(interface):
                continue
            for entry in addresses:
                if entry.family != socket.AF_INET or not is_usable_private_ipv4(entry.address):
                    continue
                candidates.append({
                    "address": entry.address,
                    "interface": interface,
                    "kind": interface_kind(interface),
                    "is_tunnel": False,
                })
    except (ImportError, OSError):
        pass

    if candidates:
        unique = {item["address"]: item for item in candidates}
        return sorted(
            unique.values(),
            key=lambda item: (
                0 if item["kind"] == "Wi-Fi" else 1 if item["kind"] == "Ethernet" else 2,
                tuple(int(part) for part in item["address"].split(".")),
            ),
        )

    candidates = set()
    try:
        for result in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            candidates.add(result[4][0])
    except OSError:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 80))
            candidates.add(probe.getsockname()[0])
    except OSError:
        pass

    addresses = []
    for candidate in candidates:
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if is_usable_private_ipv4(address):
            addresses.append({
                "address": str(address),
                "interface": "Local network",
                "kind": "Network",
                "is_tunnel": False,
            })
    return sorted(addresses, key=lambda item: tuple(int(part) for part in item["address"].split(".")))


def discover_private_ipv4_addresses():
    return [item["address"] for item in discover_private_ipv4_candidates()]


def is_usable_private_ipv4(address):
    if not isinstance(address, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
        try:
            address = ipaddress.ip_address(address)
        except ValueError:
            return False
    return (
        address.version == 4
        and address.is_private
        and not address.is_loopback
        and not address.is_link_local
    )


def build_mobile_urls(port, token, addresses=None):
    if addresses is None:
        candidates = discover_private_ipv4_candidates()
    else:
        candidates = [
            item if isinstance(item, dict) else {
                "address": str(item),
                "interface": "Local network",
                "kind": "Network",
                "is_tunnel": False,
            }
            for item in addresses
        ]
    return [
        {
            "address": item["address"],
            "interface": item.get("interface", "Local network"),
            "kind": item.get("kind", "Network"),
            "base_url": f"http://{item['address']}:{int(port)}",
            "pairing_url": f"http://{item['address']}:{int(port)}/pair#{token}",
        }
        for item in candidates
    ]


def _run_windows_network_probe(executable_path=None, port=None):
    if os.name != "nt":
        return {
            "windows": False,
            "profiles": [],
            "vpn_adapters": [],
            "firewall_blocked": False,
            "firewall_allowed": False,
        }

    script = r"""
$ErrorActionPreference = 'SilentlyContinue'
$profiles = @(Get-NetConnectionProfile | ForEach-Object {
    [pscustomobject]@{
        interface = $_.InterfaceAlias
        category = [string]$_.NetworkCategory
        ipv4 = [string]$_.IPv4Connectivity
    }
})
$vpnPattern = '(?i)(vpn|wireguard|openvpn|zerotier|hamachi|(^|[\s_-])(tun|tap)([\s_-]|$))'
$vpnAdapters = @(Get-NetAdapter | Where-Object {
    $_.Status -eq 'Up' -and (($_.Name -match $vpnPattern) -or ($_.InterfaceDescription -match $vpnPattern))
} | ForEach-Object {
    [pscustomobject]@{ name = $_.Name; description = $_.InterfaceDescription }
})
$blocked = $false
$allowed = $false
$exe = $env:SUBGEN_DIAGNOSTIC_EXE
$port = [string]$env:SUBGEN_DIAGNOSTIC_PORT
if ($exe) {
    Get-NetFirewallApplicationFilter -PolicyStore ActiveStore -Program $exe | Get-NetFirewallRule | ForEach-Object {
        $rule = $_
        if ($rule.Enabled -eq 'True' -and $rule.Direction -eq 'Inbound') {
            if ($rule.Action -eq 'Block') { $blocked = $true }
            if ($rule.Action -eq 'Allow' -and ([string]$rule.Profile -match 'Private|Any')) {
                $portFilter = $rule | Get-NetFirewallPortFilter
                if ($portFilter.Protocol -eq 'TCP' -and ($portFilter.LocalPort -eq 'Any' -or $portFilter.LocalPort -eq $port)) {
                    $allowed = $true
                }
            }
        }
    }
}
[pscustomobject]@{
    windows = $true
    profiles = $profiles
    vpn_adapters = $vpnAdapters
    firewall_blocked = $blocked
    firewall_allowed = $allowed
} | ConvertTo-Json -Depth 5 -Compress
"""
    encoded = base64.b64encode(script.encode("utf-16le")).decode("ascii")
    environment = os.environ.copy()
    if executable_path:
        environment["SUBGEN_DIAGNOSTIC_EXE"] = str(executable_path)
    if port is not None:
        environment["SUBGEN_DIAGNOSTIC_PORT"] = str(int(port))
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            capture_output=True,
            text=True,
            timeout=20,
            env=environment,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    return {
        "windows": os.name == "nt",
        "profiles": [],
        "vpn_adapters": [],
        "firewall_blocked": False,
        "firewall_allowed": False,
        "probe_failed": True,
    }


def mobile_access_diagnostics(urls, executable_path=None, port=None, windows_state=None):
    state = windows_state if windows_state is not None else _run_windows_network_probe(executable_path, port)
    profiles = {item.get("interface"): item for item in state.get("profiles", [])}
    issues = []

    if not urls:
        issues.append({
            "code": "no_lan_address",
            "severity": "blocker",
            "title": "No local network detected",
            "message": "Connect this computer and phone to the same Wi-Fi or Ethernet network.",
        })
    if state.get("probe_failed"):
        issues.append({
            "code": "diagnostics_unavailable",
            "severity": "blocker",
            "title": "Windows network check did not complete",
            "message": "SubGen could not verify the firewall and network profile. Retry the check before pairing.",
        })
    if state.get("firewall_blocked"):
        issues.append({
            "code": "firewall_blocked",
            "severity": "blocker",
            "title": "Windows Firewall is blocking SubGen",
            "message": "Repair Windows access to allow SubGen only on the private local network.",
            "repairable": True,
        })
    elif state.get("windows") and not state.get("firewall_allowed") and not state.get("probe_failed"):
        issues.append({
            "code": "firewall_not_allowed",
            "severity": "blocker",
            "title": "Windows Firewall has no SubGen access rule",
            "message": "Repair Windows access to allow SubGen only on the private local network.",
            "repairable": True,
        })

    public_interfaces = []
    for entry in urls:
        profile = profiles.get(entry.get("interface"), {})
        if str(profile.get("category", "")).lower() == "public":
            public_interfaces.append(entry.get("interface"))
    if public_interfaces:
        issues.append({
            "code": "public_network",
            "severity": "blocker",
            "title": "Wi-Fi is marked Public",
            "message": "Mark this trusted network Private so SubGen is not exposed on public Wi-Fi.",
            "repairable": True,
        })

    vpn_adapters = state.get("vpn_adapters", []) or []
    if vpn_adapters:
        names = ", ".join(sorted({item.get("name", "VPN") for item in vpn_adapters}))
        issues.append({
            "code": "vpn_detected",
            "severity": "warning",
            "title": "VPN detected",
            "message": f"{names} is active. Enable Allow LAN connections in the VPN on both this computer and the phone.",
        })

    return {
        "ready": not any(item["severity"] == "blocker" for item in issues),
        "issues": issues,
        "checked_at": int(time.time()),
    }


def request_windows_mobile_access_repair(port, interface_aliases, executable_path=None):
    if os.name != "nt":
        raise RuntimeError("Automatic firewall repair is available on Windows only.")
    executable_path = str(executable_path or sys.executable)
    aliases = [str(value) for value in interface_aliases if value]
    if not aliases:
        raise RuntimeError("No local network interface is available to repair.")

    def powershell_literal(value):
        return "'" + str(value).replace("'", "''") + "'"

    interface_values = ", ".join(powershell_literal(value) for value in aliases)
    repair_script = r"""
$ErrorActionPreference = 'Stop'
$exe = __SUBGEN_EXE__
$port = __SUBGEN_PORT__
$interfaces = @(__SUBGEN_INTERFACES__)
Get-NetFirewallApplicationFilter -PolicyStore PersistentStore -Program $exe -ErrorAction SilentlyContinue |
    Get-NetFirewallRule -ErrorAction SilentlyContinue |
    Where-Object { $_.Direction -eq 'Inbound' -and $_.Action -eq 'Block' } |
    Remove-NetFirewallRule
Get-NetFirewallRule -DisplayName 'SubGen Mobile Access (Private LAN)' -ErrorAction SilentlyContinue | Remove-NetFirewallRule
foreach ($interface in $interfaces) {
    Get-NetConnectionProfile -InterfaceAlias $interface -ErrorAction SilentlyContinue | Set-NetConnectionProfile -NetworkCategory Private
}
New-NetFirewallRule -DisplayName 'SubGen Mobile Access (Private LAN)' `
    -Description 'Allows SubGen mobile access only from the private local subnet.' `
    -Direction Inbound -Action Allow -Program $exe -Protocol TCP -LocalPort $port `
    -Profile Private -RemoteAddress LocalSubnet | Out-Null
"""
    repair_script = (
        repair_script
        .replace("__SUBGEN_EXE__", powershell_literal(executable_path))
        .replace("__SUBGEN_PORT__", str(int(port)))
        .replace("__SUBGEN_INTERFACES__", interface_values)
    )
    repair_encoded = base64.b64encode(repair_script.encode("utf-16le")).decode("ascii")
    launcher = (
        "$arguments=@('-NoProfile','-NonInteractive','-EncodedCommand','"
        + repair_encoded
        + "'); Start-Process -FilePath 'powershell.exe' -Verb RunAs -WindowStyle Hidden -ArgumentList $arguments"
    )
    environment = os.environ.copy()
    subprocess.Popen(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", launcher],
        env=environment,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        close_fds=True,
    )
