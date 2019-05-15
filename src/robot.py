import ai2thor.controller

class Robot:

    moves = {'a': 'MoveLeft', 'w': 'MoveAhead', 'd': 'MoveRight', 's': 'MoveBack'}
    rotates = {'j': -5, 'l': 5}
    looks = {'i': -5, 'k': 5}

    def __init__(self, _screen_size=600):
        self.screen_size = _screen_size
        self.rotation = 90
        self.horizon = 0
        
        self.controller = ai2thor.controller.Controller()
        self.event = None

    def start(self):
        # Kitchens: FloorPlan1 - FloorPlan30
        # Living rooms: FloorPlan201 - FloorPlan230
        # Bedrooms: FloorPlan301 - FloorPlan330
        # Bathrooms: FloorPLan401 - FloorPlan430
        
        self.controller.start(player_screen_width=self.screen_size,
                player_screen_height=self.screen_size)
        self.controller.reset('FloorPlan230')
        self.event = self.controller.step(dict(action='Initialize',
                            gridSize=0.5,
                            rotation=self.rotation,
                            horizon=self.horizon,
                            renderClassImage=False,
                            renderObjectImage=False))
    
    def getFrame(self):
        return self.event.cv2img

    def stop(self):
        self.controller.stop()

    def apply(self, action):
        if action in self.moves:
            self.event = self.controller.step(dict(action=self.moves[action]))
            
        elif action in self.rotates:
            self.rotation += self.rotates[action]
            self.event = self.controller.step(dict(action='Rotate', rotation=self.rotation))
            
        elif action in self.looks:
            self.horizon += self.looks[action]
            self.event = self.controller.step(dict(action='Look', horizon=self.horizon))
