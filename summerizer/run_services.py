#!/usr/bin/env python3
"""
Summerizer Services Runner
==========================
A comprehensive script to start, monitor, and manage all services
required for the Multimodal PDF Processing & Q&A System.

Services managed:
1. Redis (Job Queue & State Management) - Port 6379
2. Milvus (Vector Database) - Port 19530
3. Embedding Service (FastAPI) - Port 8000
4. Background Worker (RQ) - Queue: pdf-processing
5. Ollama LLM Service - Port 11434
6. Main API Server (FastAPI) - Port 8080

Usage:
    python run_services.py              # Start all services
    python run_services.py --check      # Check status only
    python run_services.py --stop       # Stop all services
    python run_services.py --restart    # Restart all services
"""

import subprocess
import sys
import os
import time
import signal
import socket
import argparse
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Configuration
# =============================================================================

class ServiceStatus(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"
    NOT_INSTALLED = "not_installed"


@dataclass
class ServiceConfig:
    name: str
    port: Optional[int]
    check_command: Optional[str]
    start_command: List[str]
    health_check_url: Optional[str] = None
    timeout: int = 30
    is_docker: bool = False
    required: bool = True


# Service configurations
SERVICES: Dict[str, ServiceConfig] = {
    "redis": ServiceConfig(
        name="Redis",
        port=6379,
        check_command="redis-cli ping",
        start_command=["redis-server", "--daemonize", "yes"],
        timeout=10,
        required=True
    ),
    "milvus": ServiceConfig(
        name="Milvus",
        port=19530,
        check_command=None,
        start_command=["docker", "start", "milvus-standalone"],
        is_docker=True,
        timeout=60,
        required=True
    ),
    "embedding": ServiceConfig(
        name="Embedding Service",
        port=8000,
        check_command=None,
        start_command=[
            sys.executable, "-m", "uvicorn",
            "embedding.embedding_service:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--workers", "1"
        ],
        health_check_url="http://localhost:8000/health",
        timeout=120,
        required=True
    ),
    "worker": ServiceConfig(
        name="RQ Worker",
        port=None,
        check_command=None,
        start_command=[sys.executable, "-m", "rq", "worker", "pdf-processing"],
        timeout=15,
        required=True
    ),
    "ollama": ServiceConfig(
        name="Ollama LLM",
        port=11434,
        check_command=None,
        start_command=["ollama", "serve"],
        health_check_url="http://localhost:11434/api/tags",
        timeout=30,
        required=False
    ),
    "main_api": ServiceConfig(
        name="Main API",
        port=8080,
        check_command=None,
        start_command=[
            sys.executable, "-m", "uvicorn",
            "document_analysis_main:app",
            "--host", "0.0.0.0",
            "--port", "8080"
        ],
        health_check_url="http://localhost:8080/health",
        timeout=30,
        required=True
    )
}

# Process tracking
running_processes: Dict[str, subprocess.Popen] = {}


# =============================================================================
# Terminal Colors and Formatting
# =============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DIM = '\033[2m'


def color(text: str, color_code: str) -> str:
    """Apply color to text."""
    return f"{color_code}{text}{Colors.ENDC}"


def print_header(text: str):
    """Print a formatted header."""
    width = 60
    print()
    print(color("=" * width, Colors.CYAN))
    print(color(f"  {text}".center(width), Colors.BOLD + Colors.CYAN))
    print(color("=" * width, Colors.CYAN))
    print()


def print_status_line(service_name: str, status: ServiceStatus, message: str = ""):
    """Print a formatted status line."""
    status_icons = {
        ServiceStatus.STOPPED: (Colors.RED, "[ STOPPED ]"),
        ServiceStatus.STARTING: (Colors.YELLOW, "[STARTING ]"),
        ServiceStatus.RUNNING: (Colors.GREEN, "[ RUNNING ]"),
        ServiceStatus.ERROR: (Colors.RED, "[  ERROR  ]"),
        ServiceStatus.NOT_INSTALLED: (Colors.YELLOW, "[NOT FOUND]"),
    }

    color_code, icon = status_icons.get(status, (Colors.DIM, "[UNKNOWN]"))
    name_padded = service_name.ljust(20)

    line = f"  {color(icon, color_code)}  {name_padded}"
    if message:
        line += f"  {color(message, Colors.DIM)}"
    print(line)


def print_info(message: str):
    """Print an info message."""
    print(f"  {color('INFO:', Colors.BLUE)} {message}")


def print_success(message: str):
    """Print a success message."""
    print(f"  {color('SUCCESS:', Colors.GREEN)} {message}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"  {color('WARNING:', Colors.YELLOW)} {message}")


def print_error(message: str):
    """Print an error message."""
    print(f"  {color('ERROR:', Colors.RED)} {message}")


# =============================================================================
# Service Status Checks
# =============================================================================

def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.settimeout(1)
            s.connect(('localhost', port))
            return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False


def check_redis_status() -> Tuple[ServiceStatus, str]:
    """Check Redis status."""
    try:
        result = subprocess.run(
            ["redis-cli", "ping"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and "PONG" in result.stdout:
            return ServiceStatus.RUNNING, "Port 6379"
        return ServiceStatus.STOPPED, ""
    except FileNotFoundError:
        return ServiceStatus.NOT_INSTALLED, "redis-cli not found"
    except Exception as e:
        return ServiceStatus.ERROR, str(e)


def check_milvus_status() -> Tuple[ServiceStatus, str]:
    """Check Milvus status."""
    try:
        # Check if docker is available
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=milvus", "--format", "{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return ServiceStatus.NOT_INSTALLED, "Docker not available"

        status_output = result.stdout.strip()

        if not status_output:
            # Container doesn't exist, try to check docker-compose
            return ServiceStatus.STOPPED, "Container not found"

        if "Up" in status_output:
            # Also verify the port is accessible
            if is_port_in_use(19530):
                return ServiceStatus.RUNNING, "Port 19530"
            return ServiceStatus.STARTING, "Container up, port not ready"

        return ServiceStatus.STOPPED, status_output
    except FileNotFoundError:
        return ServiceStatus.NOT_INSTALLED, "Docker not installed"
    except Exception as e:
        return ServiceStatus.ERROR, str(e)


def check_ollama_status() -> Tuple[ServiceStatus, str]:
    """Check Ollama status."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            return ServiceStatus.RUNNING, "Port 11434"
        return ServiceStatus.STOPPED, ""
    except ImportError:
        if is_port_in_use(11434):
            return ServiceStatus.RUNNING, "Port 11434"
        return ServiceStatus.STOPPED, ""
    except Exception:
        if is_port_in_use(11434):
            return ServiceStatus.RUNNING, "Port 11434"
        return ServiceStatus.STOPPED, ""


def check_http_service(port: int, health_url: str = None) -> Tuple[ServiceStatus, str]:
    """Check HTTP service status."""
    if not is_port_in_use(port):
        return ServiceStatus.STOPPED, ""

    if health_url:
        try:
            import requests
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200:
                return ServiceStatus.RUNNING, f"Port {port}"
        except Exception:
            pass

    return ServiceStatus.RUNNING, f"Port {port}"


def check_rq_worker_status() -> Tuple[ServiceStatus, str]:
    """Check RQ Worker status."""
    try:
        # Check if any rq worker process is running
        result = subprocess.run(
            ["pgrep", "-f", "rq worker pdf-processing"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return ServiceStatus.RUNNING, "Queue: pdf-processing"

        # Also check in our tracked processes
        if "worker" in running_processes and running_processes["worker"].poll() is None:
            return ServiceStatus.RUNNING, "Queue: pdf-processing"

        return ServiceStatus.STOPPED, ""
    except Exception as e:
        return ServiceStatus.ERROR, str(e)


def get_service_status(service_key: str) -> Tuple[ServiceStatus, str]:
    """Get the status of a specific service."""
    config = SERVICES[service_key]

    if service_key == "redis":
        return check_redis_status()
    elif service_key == "milvus":
        return check_milvus_status()
    elif service_key == "ollama":
        return check_ollama_status()
    elif service_key == "worker":
        return check_rq_worker_status()
    elif config.port:
        return check_http_service(config.port, config.health_check_url)

    return ServiceStatus.STOPPED, ""


def check_all_services() -> Dict[str, Tuple[ServiceStatus, str]]:
    """Check status of all services."""
    statuses = {}
    for key in SERVICES:
        statuses[key] = get_service_status(key)
    return statuses


# =============================================================================
# Service Management
# =============================================================================

def start_redis() -> bool:
    """Start Redis server."""
    status, _ = check_redis_status()
    if status == ServiceStatus.RUNNING:
        print_info("Redis is already running")
        return True

    if status == ServiceStatus.NOT_INSTALLED:
        print_error("Redis is not installed. Please install redis-server.")
        return False

    print_info("Starting Redis...")
    try:
        subprocess.run(
            ["redis-server", "--daemonize", "yes"],
            check=True,
            capture_output=True
        )

        # Wait for startup
        for _ in range(10):
            time.sleep(0.5)
            status, _ = check_redis_status()
            if status == ServiceStatus.RUNNING:
                print_success("Redis started successfully")
                return True

        print_error("Redis failed to start within timeout")
        return False
    except FileNotFoundError:
        print_error("Redis is not installed. Please install redis-server.")
        return False
    except Exception as e:
        print_error(f"Failed to start Redis: {e}")
        return False


def start_milvus() -> bool:
    """Start Milvus via Docker."""
    status, msg = check_milvus_status()
    if status == ServiceStatus.RUNNING:
        print_info("Milvus is already running")
        return True

    if status == ServiceStatus.NOT_INSTALLED:
        print_error("Docker is not installed or not running.")
        return False

    print_info("Starting Milvus...")
    try:
        # Try starting existing container first
        result = subprocess.run(
            ["docker", "start", "milvus-standalone"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            # Container doesn't exist, try docker-compose
            compose_file = os.path.join(os.path.dirname(__file__), "docker-compose.yml")
            if os.path.exists(compose_file):
                print_info("Starting Milvus with docker-compose...")
                subprocess.run(
                    ["docker-compose", "up", "-d", "milvus"],
                    cwd=os.path.dirname(__file__),
                    check=True,
                    capture_output=True
                )
            else:
                print_warning("No docker-compose.yml found. Attempting to pull and run Milvus...")
                # Run Milvus standalone
                subprocess.run([
                    "docker", "run", "-d",
                    "--name", "milvus-standalone",
                    "-p", "19530:19530",
                    "-p", "9091:9091",
                    "-v", f"{os.getcwd()}/milvus_data:/var/lib/milvus",
                    "milvusdb/milvus:latest",
                    "milvus", "run", "standalone"
                ], check=True, capture_output=True)

        # Wait for Milvus to be ready
        print_info("Waiting for Milvus to be ready (this may take a minute)...")
        for i in range(60):
            time.sleep(2)
            if is_port_in_use(19530):
                print_success("Milvus started successfully")
                return True
            if i % 10 == 0 and i > 0:
                print_info(f"Still waiting... ({i}s)")

        print_error("Milvus failed to start within timeout")
        return False
    except Exception as e:
        print_error(f"Failed to start Milvus: {e}")
        return False


def start_background_service(service_key: str, log_dir: str = "logs") -> bool:
    """Start a background service (embedding, worker, main_api)."""
    config = SERVICES[service_key]
    status, _ = get_service_status(service_key)

    if status == ServiceStatus.RUNNING:
        print_info(f"{config.name} is already running")
        return True

    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{service_key}.log")

    print_info(f"Starting {config.name}...")
    try:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                config.start_command,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                start_new_session=True
            )

        running_processes[service_key] = process

        # Wait for service to be ready
        timeout = config.timeout
        for i in range(timeout):
            time.sleep(1)

            # Check if process is still running
            if process.poll() is not None:
                print_error(f"{config.name} exited unexpectedly. Check {log_file}")
                return False

            status, _ = get_service_status(service_key)
            if status == ServiceStatus.RUNNING:
                print_success(f"{config.name} started successfully")
                return True

            if i % 10 == 0 and i > 0:
                print_info(f"Waiting for {config.name}... ({i}s)")

        print_warning(f"{config.name} started but health check timed out. Check {log_file}")
        return True  # Process is running, just health check timeout

    except Exception as e:
        print_error(f"Failed to start {config.name}: {e}")
        return False


def start_ollama() -> bool:
    """Start Ollama service."""
    status, _ = check_ollama_status()
    if status == ServiceStatus.RUNNING:
        print_info("Ollama is already running")
        return True

    print_info("Starting Ollama...")
    try:
        # Check if ollama is installed
        result = subprocess.run(
            ["which", "ollama"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print_warning("Ollama is not installed. Q&A features will not work.")
            print_info("Install from: https://ollama.ai")
            return False

        # Start ollama serve in background
        log_file = os.path.join("logs", "ollama.log")
        os.makedirs("logs", exist_ok=True)

        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True
            )

        running_processes["ollama"] = process

        # Wait for startup
        for i in range(30):
            time.sleep(1)
            status, _ = check_ollama_status()
            if status == ServiceStatus.RUNNING:
                print_success("Ollama started successfully")

                # Check for required model
                check_ollama_model()
                return True

        print_warning("Ollama started but not responding. Check logs/ollama.log")
        return True

    except Exception as e:
        print_warning(f"Failed to start Ollama: {e}")
        return False


def check_ollama_model():
    """Check if required Ollama model is installed."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if "gemma3:4b" not in result.stdout and "gemma3" not in result.stdout:
            print_warning("Model 'gemma3:4b' not found. Pull it with: ollama pull gemma3:4b")
    except Exception:
        pass


def stop_service(service_key: str) -> bool:
    """Stop a specific service."""
    config = SERVICES[service_key]

    if service_key == "redis":
        try:
            subprocess.run(["redis-cli", "shutdown"], capture_output=True, timeout=5)
            print_success(f"Stopped {config.name}")
            return True
        except Exception:
            pass

    elif service_key == "milvus":
        try:
            subprocess.run(["docker", "stop", "milvus-standalone"], capture_output=True, timeout=30)
            print_success(f"Stopped {config.name}")
            return True
        except Exception:
            pass

    elif service_key in running_processes:
        process = running_processes[service_key]
        if process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=10)
                print_success(f"Stopped {config.name}")
                return True
            except Exception:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    print_success(f"Force stopped {config.name}")
                    return True
                except Exception:
                    pass

    # Try to kill by port
    if config.port:
        try:
            result = subprocess.run(
                ["lsof", "-t", f"-i:{config.port}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                    except Exception:
                        pass
                print_success(f"Stopped {config.name}")
                return True
        except Exception:
            pass

    # Kill by process name for worker
    if service_key == "worker":
        try:
            subprocess.run(
                ["pkill", "-f", "rq worker pdf-processing"],
                capture_output=True,
                timeout=5
            )
            print_success(f"Stopped {config.name}")
            return True
        except Exception:
            pass

    return False


def stop_all_services():
    """Stop all services."""
    print_header("Stopping All Services")

    # Stop in reverse order
    for key in reversed(list(SERVICES.keys())):
        status, _ = get_service_status(key)
        if status == ServiceStatus.RUNNING:
            stop_service(key)
        else:
            print_info(f"{SERVICES[key].name} is not running")


# =============================================================================
# Main Functions
# =============================================================================

def display_status():
    """Display status of all services."""
    print_header("Service Status")

    statuses = check_all_services()

    for key, config in SERVICES.items():
        status, msg = statuses[key]
        print_status_line(config.name, status, msg)

    print()

    # Summary
    running_count = sum(1 for s, _ in statuses.values() if s == ServiceStatus.RUNNING)
    required_count = sum(1 for k, c in SERVICES.items() if c.required)
    required_running = sum(1 for k, c in SERVICES.items()
                          if c.required and statuses[k][0] == ServiceStatus.RUNNING)

    if required_running == required_count:
        print_success(f"All required services running ({running_count}/{len(SERVICES)} total)")
        print()
        print(f"  {color('Web UI:', Colors.BOLD)} http://localhost:8080")
        print(f"  {color('API Docs:', Colors.BOLD)} http://localhost:8080/docs")
    else:
        print_warning(f"{required_running}/{required_count} required services running")


def start_all_services():
    """Start all services in order."""
    print_header("Starting Summerizer Services")

    print(f"  {color('Working Directory:', Colors.DIM)} {os.getcwd()}")
    print()

    # Check dependencies
    print_info("Checking dependencies...")

    success = True

    # 1. Start Redis
    print()
    print(color("  [1/6] Redis (Job Queue)", Colors.BOLD))
    if not start_redis():
        print_error("Redis is required. Cannot continue.")
        return False

    # 2. Start Milvus
    print()
    print(color("  [2/6] Milvus (Vector Database)", Colors.BOLD))
    if not start_milvus():
        print_error("Milvus is required. Cannot continue.")
        return False

    # 3. Start Embedding Service
    print()
    print(color("  [3/6] Embedding Service", Colors.BOLD))
    if not start_background_service("embedding"):
        print_error("Embedding service is required. Cannot continue.")
        return False

    # 4. Start RQ Worker
    print()
    print(color("  [4/6] Background Worker", Colors.BOLD))
    if not start_background_service("worker"):
        print_error("RQ Worker is required. Cannot continue.")
        return False

    # 5. Start Ollama (optional)
    print()
    print(color("  [5/6] Ollama LLM (Optional)", Colors.BOLD))
    start_ollama()  # Optional, don't fail if not available

    # 6. Start Main API
    print()
    print(color("  [6/6] Main API Server", Colors.BOLD))
    if not start_background_service("main_api"):
        print_error("Main API is required. Cannot continue.")
        return False

    # Final status
    print()
    display_status()

    return True


def wait_for_shutdown():
    """Wait for Ctrl+C and cleanup."""
    print()
    print(color("  Press Ctrl+C to stop all services", Colors.DIM))
    print()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print()
        stop_all_services()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Summerizer Services Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_services.py              Start all services
  python run_services.py --check      Check status only
  python run_services.py --stop       Stop all services
  python run_services.py --restart    Restart all services
        """
    )

    parser.add_argument(
        '--check', '-c',
        action='store_true',
        help='Check status of all services'
    )
    parser.add_argument(
        '--stop', '-s',
        action='store_true',
        help='Stop all services'
    )
    parser.add_argument(
        '--restart', '-r',
        action='store_true',
        help='Restart all services'
    )
    parser.add_argument(
        '--no-wait',
        action='store_true',
        help='Exit after starting services (do not wait for Ctrl+C)'
    )

    args = parser.parse_args()

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print()
    print(color("  Summerizer - Multimodal PDF Processing & Q&A System", Colors.BOLD + Colors.CYAN))
    print(color("  " + "=" * 50, Colors.CYAN))

    if args.check:
        display_status()
    elif args.stop:
        stop_all_services()
    elif args.restart:
        stop_all_services()
        time.sleep(2)
        if start_all_services() and not args.no_wait:
            wait_for_shutdown()
    else:
        if start_all_services() and not args.no_wait:
            wait_for_shutdown()


if __name__ == "__main__":
    main()
