---
domain: it
category: Network & VPN
title: VPN Setup & Troubleshooting
---
# VPN Setup & Troubleshooting

## Who needs the VPN
You need the NovaTech VPN to reach internal systems (Jenkins, staging environments, the finance ERP and internal wikis) when you are not on the office network. Email, Slack, PeopleHub and Google Drive work without the VPN.

## Supported client
NovaTech uses **GlobalProtect** (Palo Alto Networks). The portal address is `vpn.novatech.com`. Company-managed laptops already have GlobalProtect installed through the device-management tool. Personal devices are not allowed to connect to the VPN.

## Connecting on Windows or macOS
1. Click the GlobalProtect icon in the system tray (Windows) or the menu bar (macOS).
2. If asked for a portal, enter `vpn.novatech.com`.
3. Sign in with your SSO username and password.
4. Approve the push notification in the Okta Verify app.
5. The icon turns into a shield with a check mark when you are connected.

## Connecting on Linux (engineering laptops)
Install the `globalprotect` package from the internal repository with `sudo apt install globalprotect`, then run `globalprotect connect --portal vpn.novatech.com` and complete the Okta Verify prompt.

## Common errors and fixes
- **"Gateway not responding"**: switch networks (for example, from hotel Wi-Fi to a phone hotspot); some public networks block VPN traffic.
- **Authentication failed**: your password may have expired. Reset it at https://passwords.novatech.com and retry after five minutes.
- **Connected but internal sites do not load**: disconnect, run `ipconfig /flushdns` on Windows (or `sudo dscacheutil -flushcache` on macOS), and reconnect.
- **No Okta push received**: open Okta Verify and pull to refresh, or choose "Enter code" instead.

## Escalation
If none of the fixes work, raise a ticket with the IT Service Desk and include your laptop asset tag, the exact error message and the network you are on.
