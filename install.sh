#!/bin/bash

# This script is used to install the Shake&Tune module on a Klipper machine


USER_CONFIG_PATH="${HOME}/printer_data/config"
MOONRAKER_CONFIG="${HOME}/printer_data/config/moonraker.conf"
KLIPPER_PATH="${HOME}/klipper"
KLIPPER_VENV_PATH="${HOME}/klippy-env"

OLD_K_SHAKETUNE_VENV="${HOME}/klippain_shaketune-env"
K_SHAKETUNE_PATH="${HOME}/klippain_shaketune"

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

    if sudo systemctl is-active --quiet klipper; then
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
    packages=("libopenblas-dev" "libatlas-base-dev" "sudo")
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

function setup_shaketune_sudo {
    echo "[SETUP] Setting up sudo permissions for Shake&Tune swap file management..."
    chmod +x ${K_SHAKETUNE_PATH}/install_swap_access.sh
    ${K_SHAKETUNE_PATH}/install_swap_access.sh
    printf "\n"
}

function setup_venv {
    if [ ! -d "${KLIPPER_VENV_PATH}" ]; then
        echo "[ERROR] Klipper's Python virtual environment not found!"
        exit -1
    fi

    if [ -d "${OLD_K_SHAKETUNE_VENV}" ]; then
        echo "[INFO] Old K-Shake&Tune virtual environment found, cleaning it!"
        rm -rf "${OLD_K_SHAKETUNE_VENV}"
    fi

    source "${KLIPPER_VENV_PATH}/bin/activate"
    echo "[SETUP] Installing/Updating K-Shake&Tune dependencies..."
    pip install --upgrade pip
    pip install -r "${K_SHAKETUNE_PATH}/requirements.txt"
    deactivate
    printf "\n"
}

function link_extension {
    # Reusing the old linking extension function to cleanup and remove the macros for older S&T versions

    if [ -d "${HOME}/klippain_config" ] && [ -f "${USER_CONFIG_PATH}/.VERSION" ]; then
        if [ -d "${USER_CONFIG_PATH}/scripts/K-ShakeTune" ]; then
            echo "[INFO] Old K-Shake&Tune macro folder found, cleaning it!"
            rm -rf "${USER_CONFIG_PATH}/scripts/K-ShakeTune"
        fi
    else
        if [ -d "${USER_CONFIG_PATH}/K-ShakeTune" ]; then
            echo "[INFO] Old K-Shake&Tune macro folder found, cleaning it!"
            rm -rf "${USER_CONFIG_PATH}/K-ShakeTune"
        fi
    fi
}

function link_module {
    if [ ! -d "${KLIPPER_PATH}/klippy/extras/shaketune" ]; then
        echo "[INSTALL] Linking Shake&Tune module to Klipper extras"
        ln -frsn ${K_SHAKETUNE_PATH}/shaketune ${KLIPPER_PATH}/klippy/extras/shaketune
    else
        printf "[INSTALL] Klippain Shake&Tune Klipper module is already installed. Continuing...\n\n"
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
setup_shaketune_sudo
setup_venv
link_extension
link_module
add_updater


echo "[POST-INSTALL] Shake&Tune installation complete!"

# Ask the user if he want to reboot now
read -p "Do you want to reboot now? [y/N] " answer1
case $answer1 in
    [yY][eE][sS]|[yY])
        sudo reboot
        ;;
    *)
        # Ask the user if he still want to restart Klipper and Moonraker
        read -p "Ok, but do you want to at least restart Klipper and Moonraker? [y/N] " answer2
        case $answer2 in
            [yY][eE][sS]|[yY])
                restart_klipper
                restart_moonraker
                ;;
            *)
            echo "Ok, but just a heads-up: Shake&Tune won't be available until you restart your printer!"
        esac
        ;;        
esac
