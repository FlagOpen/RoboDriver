from dora import Node

node = Node()

def main():
    for event in node:
        if event["type"] == "INPUT":
            if "joint" in event["id"]:
                data = event["value"].to_numpy()
                print(f"Node print_joint: recieved dataflow {event["id"]}: {data}")

        if event["type"] == "STOP":
            break

if __name__ == "__main__":
    main()