from dora import Node


node = Node()

def main():
    for event in node:
        if event["type"] == "INPUT":
            if "jointstate" in event["id"]:
                data = event["value"].to_numpy()
                print(f"Dora node recieved dataflow \"{event["id"]}\": \n{data}")

        if event["type"] == "STOP":
            break

if __name__ == "__main__":
    main()