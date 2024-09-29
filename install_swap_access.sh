#!/bin/bash

# Sets up sudo permissions for the shaketune module to allow
# the user to create and delete a swap file without requiring a password


set -e

SUDOERS_DIR='/etc/sudoers.d'
SUDOERS_FILE='020-sudo-for-shaketune'
NEW_GROUP='shaketunesudo'


verify_ready() {
    if [ "$EUID" -eq 0 ]; then
        echo "This script must not run as root"
        exit -1
    fi
}

create_sudoers_file() {
    SCRIPT_TEMP_PATH=/tmp

    echo "Creating ${SUDOERS_FILE} ..."
    sudo rm -f $SCRIPT_TEMP_PATH/$SUDOERS_FILE
    sudo tee $SCRIPT_TEMP_PATH/$SUDOERS_FILE > /dev/null << EOF
Cmnd_Alias SWAP_CREATE = /usr/bin/fallocate -l * /home/*/shaketune_swap, /bin/dd if=/dev/zero of=/home/*/shaketune_swap bs=* count=*
Cmnd_Alias SWAP_SETUP = /sbin/mkswap /home/*/shaketune_swap, /sbin/swapon /home/*/shaketune_swap
Cmnd_Alias SWAP_REMOVE = /sbin/swapoff /home/*/shaketune_swap, /bin/rm /home/*/shaketune_swap

%${NEW_GROUP} ALL=(root) NOPASSWD: SWAP_CREATE, SWAP_SETUP, SWAP_REMOVE
EOF
}

verify_syntax() {
    if command -v visudo &> /dev/null; then
        echo "Verifying syntax of ${SUDOERS_FILE}..."
        if sudo visudo -cf $SCRIPT_TEMP_PATH/$SUDOERS_FILE; then
            VERIFY_STATUS=0
            echo "Syntax OK"
        else
            echo "Syntax Error: Check file at $SCRIPT_TEMP_PATH/$SUDOERS_FILE"
            exit 1
        fi
    else
        VERIFY_STATUS=0
        echo "Command 'visudo' not found. Skipping syntax verification."
    fi
}

install_sudoers_file() {
    verify_syntax
    if [ $VERIFY_STATUS -eq 0 ]; then
        echo "Installing sudoers file..."
        sudo chmod 0440 $SCRIPT_TEMP_PATH/$SUDOERS_FILE
        sudo cp $SCRIPT_TEMP_PATH/$SUDOERS_FILE $SUDOERS_DIR/$SUDOERS_FILE
    else
        exit 1
    fi
}

add_new_group() {
    if ! getent group $NEW_GROUP &> /dev/null; then
        echo "Creating group ${NEW_GROUP}..."
        sudo groupadd --system $NEW_GROUP
    else
        echo "Group ${NEW_GROUP} already exists."
    fi
}

add_user_to_group() {
    if groups $USER | grep -qw $NEW_GROUP; then
        echo "User ${USER} is already in group ${NEW_GROUP}."
    else
        echo "Adding user ${USER} to group ${NEW_GROUP}..."
        sudo usermod -aG $NEW_GROUP $USER
    fi
}

clean_temp() {
    sudo rm -f $SCRIPT_TEMP_PATH/$SUDOERS_FILE
}


# Run steps
verify_ready
create_sudoers_file
install_sudoers_file
add_new_group
add_user_to_group
clean_temp

exit 0
