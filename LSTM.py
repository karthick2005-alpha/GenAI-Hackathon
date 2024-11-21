def generate_output(character_line):
    # Check the character speaking and generate a response based on it
    if "ROMEO" in character_line:
        return "Romeo speaks passionately about his love for Juliet, lamenting the challenges they face."
    elif "JULIET" in character_line:
        return "Juliet expresses her feelings of hope and longing for Romeo despite the odds."
    else:
        return "The character shares a reflection or a thought on the situation."

def main():
    # Sample input from the user
    input_line = input("Enter a line (e.g., ROMEO: ): ")

    # Generate and print the response
    output = generate_output(input_line)
    print(output)

# Run the main function to prompt for input and show output
if __name__ == "__main__":
    main()