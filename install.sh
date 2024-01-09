#!/bin/bash

USER_CONFIG_PATH="${HOME}/printer_data/config"
MOONRAKER_CONFIG="${HOME}/printer_data/config/moonraker.conf"
KLIPPER_PATH="${HOME}/klipper"

K_SHAKETUNE_PATH="${HOME}/klippain_shaketune"
K_SHAKETUNE_VENV_PATH="${HOME}/klippain_shaketune-env"

set -eu
export LC_ALL=C


function preflight_checks {
    if [ "$EUID" -eq 0 ]; then
        echo "[PRE-CHECK] This script must not be run as root!"
        exit -1
    fi

    if ! command -v python3 &> /dev/null; then
        echo "[ERROR] Python 3 is not installed. Please install Python 3 to use the Shake&Tune module!"
        exit -1
    fi

    if [ "$(sudo systemctl list-units --full -all -t service --no-legend | grep -F 'klipper.service')" ]; then
        printf "[PRE-CHECK] Klipper service found! Continuing...\n\n"
    else
        echo "[ERROR] Klipper service not found, please install Klipper first!"
        exit -1
    fi

    install_package_requirements
}

# Function to check if a package is installed
function is_package_installed {
    dpkg -s "$1" &> /dev/null
    return $?
}

function install_package_requirements {
    packages=("python3-venv" "libopenblas-dev" "libatlas-base-dev")
    packages_to_install=""

    for package in "${packages[@]}"; do
        if is_package_installed "$package"; then
            echo "$package is already installed"
        else
            packages_to_install="$packages_to_install $package"
        fi
    done

    if [ -n "$packages_to_install" ]; then
        echo "Installing missing packages: $packages_to_install"
        sudo apt update && sudo apt install -y $packages_to_install
    fi
}

function check_download {
    local shaketunedirname shaketunebasename
    shaketunedirname="$(dirname ${K_SHAKETUNE_PATH})"
    shaketunebasename="$(basename ${K_SHAKETUNE_PATH})"

    if [ ! -d "${K_SHAKETUNE_PATH}" ]; then
        echo "[DOWNLOAD] Downloading Klippain Shake&Tune module repository..."
        if git -C $shaketunedirname clone https://github.com/Frix-x/klippain-shaketune.git $shaketunebasename; then
            chmod +x ${K_SHAKETUNE_PATH}/install.sh
            printf "[DOWNLOAD] Download complete!\n\n"
        else
            echo "[ERROR] Download of Klippain Shake&Tune module git repository failed!"
            exit -1
        fi
    else
        printf "[DOWNLOAD] Klippain Shake&Tune module repository already found locally. Continuing...\n\n"
    fi
}

function setup_venv {
    if [ ! -d "${K_SHAKETUNE_VENV_PATH}" ]; then
        echo "[SETUP] Creating Python virtual environment..."
        python3 -m venv "${K_SHAKETUNE_VENV_PATH}"
    else
        echo "[SETUP] Virtual environment already exists. Continuing..."
    fi

    source "${K_SHAKETUNE_VENV_PATH}/bin/activate"
    echo "[SETUP] Installing/Updating K-Shake&Tune dependencies..."
    pip install --upgrade pip
    pip install -r "${K_SHAKETUNE_PATH}/requirements.txt"
    deactivate
    printf "\n"
}

function link_extension {
    echo "[INSTALL] Linking scripts to your config directory..."

    if [ -d "${HOME}/klippain_config" ] && [ -f "${USER_CONFIG_PATH}/.VERSION" ]; then
        echo "[INSTALL] Klippain full installation found! Linking module to the script folder of Klippain"
        ln -frsn ${K_SHAKETUNE_PATH}/K-ShakeTune ${USER_CONFIG_PATH}/scripts/K-ShakeTune
    else
        ln -frsn ${K_SHAKETUNE_PATH}/K-ShakeTune ${USER_CONFIG_PATH}/K-ShakeTune
    fi
}

function link_gcodeshellcommandpy {
    if [ ! -f "${KLIPPER_PATH}/klippy/extras/gcode_shell_command.py" ]; then
        echo "[INSTALL] Downloading gcode_shell_command.py Klipper extension needed for this module"
        wget -P ${KLIPPER_PATH}/klippy/extras https://raw.githubusercontent.com/Frix-x/klippain/main/scripts/gcode_shell_command.py
    else
        printf "[INSTALL] gcode_shell_command.py Klipper extension is already installed. Continuing...\n\n"
    fi
}

function add_updater {
    update_section=$(grep -c '\[update_manager[a-z ]* Klippain-ShakeTune\]' $MOONRAKER_CONFIG || true)
    if [ "$update_section" -eq 0 ]; then
        echo -n "[INSTALL] Adding update manager to moonraker.conf..."
        cat ${K_SHAKETUNE_PATH}/moonraker.conf >> $MOONRAKER_CONFIG
    fi
}

function restart_klipper {
    echo "[POST-INSTALL] Restarting Klipper..."
    sudo systemctl restart klipper
}

function restart_moonraker {
    echo "[POST-INSTALL] Restarting Moonraker..."
    sudo systemctl restart moonraker
}


printf "\n=============================================\n"
echo "- Klippain Shake&Tune module install script -"
printf "=============================================\n\n"


# Run steps
preflight_checks
check_download
setup_venv
link_extension
add_updater
link_gcodeshellcommandpy
restart_klipper
restart_moonraker
