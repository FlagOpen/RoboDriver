from dora import Node

node = Node()

def main():
    for event in node:
        if event["type"] == "INPUT":
            if event["id"] == "joint":
                data = event["value"].to_numpy()
                print(f"Node print_joint: recieved dataflow \"joint\": {data}")

        if event["type"] == "STOP":
            break

if __name__ == "__main__":
    main()