def write_to_file(filename, content):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(content + '\n')
    print(f"{content}\n")

def clear_file(filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("")