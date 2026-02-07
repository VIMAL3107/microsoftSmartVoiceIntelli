
import socket
import sys

def check_connection(host, port, timeout=5):
    try:
        print(f"Testing connection to {host}:{port}...")
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        print(f"[SUCCESS] Connected to {host}:{port}")
        return True
    except OSError as e:
        print(f"[FAILURE] Could not connect to {host}:{port}. Error: {e}")
        return False

def main():
    print("--- Network Connectivity Test ---")
    
    # 1. Check general internet connectivity (Google DNS)
    print("\n1. Checking Internet Access (8.8.8.8:53)...")
    if not check_connection("8.8.8.8", 53):
        print("CRITICAL: Cannot reach the internet. Please check your network connection.")
    
    # 2. Check DNS resolution for smtp.gmail.com
    print("\n2. Checking DNS Resolution for smtp.gmail.com...")
    try:
        ip = socket.gethostbyname("smtp.gmail.com")
        print(f"[SUCCESS] smtp.gmail.com resolved to {ip}")
    except socket.gaierror as e:
        print(f"[FAILURE] DNS resolution failed for smtp.gmail.com. Error: {e}")
    
    # 3. Check SMTP Ports
    print("\n3. Checking SMTP Ports...")
    check_connection("smtp.gmail.com", 587)
    check_connection("smtp.gmail.com", 465)
    check_connection("smtp.gmail.com", 25)

if __name__ == "__main__":
    main()
