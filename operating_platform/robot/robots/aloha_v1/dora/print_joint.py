from dora import Node

node = Node()


def main():
    for event in node:
        if event["type"] == "INPUT":
            if event["id"] == "jointstate_piper_left":
                data = event["value"].to_numpy()
                print(f"recieved data from jointstate_piper_left: {data}")
            if event["id"] == "jointstate_piper_right":
                data = event["value"].to_numpy()
                print(f"recieved data from jointstate_piper_right: {data}")

        if event["type"] == "STOP":
            break

if __name__ == "__main__":
    main()