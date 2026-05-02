def write_to_file(filename, content):
    with open(filename, "a") as f:
        f.write(content + '\n')
    print(f"{content}\n")

def clear_file(filename):
    with open(filename, "w") as f:
        f.write("")