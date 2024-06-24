import cv2
from ultralytics import YOLO
import supervision as sv
import pygame
import time
import random

items_choice = ['Rock', 'Paper', 'Scissors']
counter1 = 0
counter2 = 0


def get_choice(src_list):
    return random.choice(src_list)


def get_result(item1, item2, counter1, counter2):
    if item1 == 'Rock' and item2 == 'Scissors':
        counter1 += 1
    elif item1 == 'Scissors' and item2 == 'Rock':
        counter2 += 1
    elif item1 == 'Paper' and item2 == 'Scissors':
        counter2 += 1
    elif item1 == 'Scissors' and item2 == 'Paper':
        counter1 += 1
    elif item1 == 'Rock' and item2 == 'Paper':
        counter2 += 1
    elif item1 == 'Paper' and item2 == 'Rock':
        counter1 += 1
    return counter1, counter2


cap = cv2.VideoCapture(0)

model = YOLO("runs/detect/train2/weights/best.pt")
box = sv.BoxAnnotator(
    thickness=1,
    text_thickness=1,
    text_scale=1
)

pygame.init()

background_colour = (255, 255, 255)
screen_width = 600
screen_height = 300
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Rock Paper Scissors with YOLOv8")
screen.fill(background_colour)
pygame.display.flip()

font = pygame.font.Font(None, 36)
countdown_time = 3
start_time = None
timer_event = pygame.USEREVENT + 1
pygame.time.set_timer(timer_event, 1000)

button_colour = (100, 100, 100)
button_text = font.render("Start", True, (255, 255, 255))
button_rect = button_text.get_rect(center=(screen_width // 2, screen_height - 30))

item = ''
item2 = ''

score_font = pygame.font.Font(None, 36)

running = True
while True:
    ret, frame = cap.read()
    result = model(frame)[0]
    detections = sv.Detections.from_yolov8(result).with_nms(threshold=0.5)
    labels = [
        f"{model.model.names[class_id]}"
        for _, _, _, class_id, _ in detections
    ]
    # print(labels)
    frame = box.annotate(
        scene=frame,
        detections=detections,
        labels=labels
    )
    cv2.imshow('yolo', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == timer_event:
            if start_time is not None:
                current_time = time.time()
                elapsed_time = current_time - start_time
                remaining_time = max(0, countdown_time - int(elapsed_time))

                pygame.draw.rect(screen, background_colour, (0, 0, screen_width, 50))

                timer_text = font.render(f"Time: {remaining_time}", True, (0, 0, 0))
                screen.blit(timer_text, (3, 3))

                pygame.display.flip()

                if remaining_time == 0:
                    start_time = None

                    item = get_choice(items_choice)

                    if len(labels) == 0:
                        break

                    item2 = labels[0]

                    counter1, counter2 = get_result(item, item2, counter1, counter2)

                    if counter1 == 3:
                        win_text = score_font.render("Machine Wins!", True, (255, 0, 0))
                        win_rect = win_text.get_rect(center=(screen_width // 2 + 60, screen_height // 2))
                        screen.blit(win_text, win_rect.topleft)
                        pygame.display.flip()
                    elif counter2 == 3:
                        win_text = score_font.render("You Win!", True, (255, 0, 0))
                        win_rect = win_text.get_rect(center=(screen_width // 2 + 60, screen_height // 2))
                        screen.blit(win_text, win_rect.topleft)
                        pygame.display.flip()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if button_rect.collidepoint(event.pos):
                start_time = time.time()
                item = ''
                item2 = ''
                if counter1 == 3 or counter2 == 3:
                    counter1 = 0
                    counter2 = 0

                pygame.draw.rect(screen, background_colour, (screen_width // 2 - 100, screen_height // 2 - 25, 300, 50))

    pygame.draw.rect(screen, button_colour, button_rect)
    screen.blit(button_text, button_rect.topleft)

    pygame.draw.rect(screen, background_colour, (0, 100, 300, 50))
    value_text = font.render(f"Machine: {item}", True, (0, 0, 0))
    screen.blit(value_text, (10, 100))

    pygame.draw.rect(screen, background_colour, (0, 150, 300, 50))
    value_text2 = font.render(f"Person: {item2}", True, (0, 0, 0))
    screen.blit(value_text2, (10, 150))

    score_text = score_font.render(f"Score: {counter1} - {counter2}", True, (0, 0, 0))
    score_rect = score_text.get_rect(topright=(screen_width - 10, 10))
    screen.blit(score_text, score_rect)

    pygame.display.flip()

pygame.quit()
cv2.destroyAllWindows()
