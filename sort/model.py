import cv2
import csv
import numpy as np

def track_people(video_path):
    
    # Load YOLOv4
    net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
    output_layers = net.getUnconnectedOutLayersNames()

    # Load video
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize variables
    people = {}
    frame_counter = 0

    # Create CSV file for output
    output_filename = 'tracking_results.csv'
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Person ID', 'Entry Frame', 'Exit Frame', 'Duration'])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_counter += 1

            # Detect people using YOLOv4
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (608, 608), swapRB=True, crop=False)
            net.setInput(blob)
            detections = net.forward(output_layers)

            # Process detections
            boxes = []
            confidences = []
            class_ids = []

            for detection in detections:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:  # Class ID 0 represents 'person'
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)
                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

            # Apply non-maximum suppression to remove overlapping detections
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Process each person detection
            for i in indices:
                i = i[0]
                box = boxes[i]
                x, y, width, height = box
                center_x = x + width // 2
                center_y = y + height // 2
                confidence = confidences[i]
                class_id = class_ids[i]

                # Assign unique ID to each person
                if class_id == 0:
                    person_id = f'{frame_counter}_{i}'  # Unique ID based on frame and index
                    if person_id not in people:
                        people[person_id] = {
                            'entry_frame': frame_counter,
                            'exit_frame': None,
                            'duration': None
                        }

                # Draw bounding box and ID on the frame
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, person_id, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame with bounding boxes and IDs
            cv2.imshow('Tracking', frame)

            # Write data to CSV file
            for person_id, person_info in people.items():
                if person_info['exit_frame'] is None:
                    person_info['exit_frame'] = frame_counter
                    person_info['duration'] = person_info['exit_frame'] - person_info['entry_frame'] + 1
                    writer.writerow([person_id, person_info['entry_frame'], person_info['exit_frame'], person_info['duration'] / fps])

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = './IMG_0(1).mp4'

    track_people(video_path)
