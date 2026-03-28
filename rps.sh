#!/bin/bash
# Rock Paper Scissors, you against the computer.
# Your Name

computer_choice=-1
user_choice=-1

compute_computer_choice () {
	computer_choice=$(( ( RANDOM % 3 ) + 1 ))
	if (( computer_choice == 1 )); then
		echo "Computer chose 1. Rock"
	elif (( computer_choice == 2 )); then
		echo "Computer chose 2. Paper"
	elif (( computer_choice == 3 )); then
		echo "Computer chose 3. Scissors"
	fi
}

print_user_choice () {
	if (( user_choice == 1 )); then
		echo "You chose 1. Rock"
	elif (( user_choice == 2 )); then
		echo "You chose 2. Paper"
	elif (( user_choice == 3 )); then
		echo "You chose 3. Scissors"
	fi
}

compute_winner () {
	if (( user_choice == computer_choice )); then
		echo "You tied.  Try again."
	elif (( user_choice == 1 && computer_choice == 3 )); then
		echo "You win."
	elif (( user_choice == 2 && computer_choice == 1 )); then
		echo "You win."
	elif (( user_choice == 3 && computer_choice == 2 )); then
		echo "You win."
	else
		echo "You lost."
	fi
}

while true; do
	echo ""
	echo "1. Rock"
	echo "2. Paper"
	echo "3. Scissors"
	echo "4. Done"
	echo -n "Your choice [1-4]? "
	read user_choice

	if (( user_choice == 4 )); then
		break
	fi

	echo ""
	print_user_choice
	compute_computer_choice
	compute_winner

done
echo "Bye"
