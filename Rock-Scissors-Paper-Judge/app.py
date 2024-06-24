import pygame
import time
import random


class App:
    def __init__(self):
        self.labels = []
        self.items_choice = ['Rock', 'Paper', 'Scissors']
        self.game_over = True

    def set_labels(self, labels):
        self.labels = labels

    def get_labels(self):
        return self.labels

    def get_items_choice(self):
        return random.choice(self.items_choice)

    def get_game_over(self):
        return self.game_over

    def start_game(self):
        # Initialize Pygame
        pygame.init()

        # Set up the screen
        background_colour = (255, 255, 255)
        screen_width = 600
        screen_height = 300
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption('Game')
        screen.fill(background_colour)
        pygame.display.flip()

        # Countdown timer settings
        font = pygame.font.Font(None, 36)  # Choose font and size
        countdown_time = 3  # Countdown time in seconds
        start_time = None  # Will be set when the "Start Over" button is clicked
        timer_event = pygame.USEREVENT + 1  # Custom event for the timer
        pygame.time.set_timer(timer_event, 1000)  # Timer event every second

        # Button settings
        button_colour = (100, 100, 100)
        button_text = font.render("Start", True, (255, 255, 255))
        button_rect = button_text.get_rect(center=(screen_width // 2, screen_height - 30))

        item = ''
        item2 = ''

        # Main loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == timer_event:
                    if start_time is not None:  # Update timer only if start_time is set
                        # Calculate remaining time
                        current_time = time.time()
                        elapsed_time = current_time - start_time
                        remaining_time = max(0, countdown_time - int(elapsed_time))

                        # Clear the previous timer text
                        pygame.draw.rect(screen, background_colour, (0, 0, screen_width, 50))

                        # Render and blit the countdown timer text
                        timer_text = font.render(f"Time: {remaining_time}", True, (0, 0, 0))
                        screen.blit(timer_text, (3, 3))

                        # Update the display
                        pygame.display.flip()

                        if remaining_time == 0:
                            # Reset the timer
                            start_time = None
                            # Get a new random item
                            item = self.get_items_choice()
                            item2 = self.get_labels()
                            self.game_over = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if button_rect.collidepoint(event.pos):  # Check if the mouse click is within the button rectangle
                        # Reset the timer
                        start_time = time.time()
                        # Сбрасываем значения при нажатии кнопки
                        item = ''
                        item2 = ''

            # Draw the button
            pygame.draw.rect(screen, button_colour, button_rect)
            screen.blit(button_text, button_rect.topleft)

            pygame.draw.rect(screen, background_colour, (0, 100, 200, 50))
            # Draw the value from getRand() on the left side of the screen
            value_text = font.render(f"Value: {item}", True, (0, 0, 0))
            screen.blit(value_text, (10, 100))

            # Draw the value2 from getRand() under the Value
            value_text2 = font.render(f"Value2: {item2}", True, (0, 0, 0))
            screen.blit(value_text2, (10, 150))

            # Update the display
            pygame.display.flip()

        # Quit Pygame
        pygame.quit()