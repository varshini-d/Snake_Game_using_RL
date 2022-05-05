Project Overview:
 
The goal of the project was to develop agents to play the classic snake game while comparing the performance of
different algorithms. In summary, we used A2C, DQN and PPO algorithms to compare which one performs the best and the 
agents use a deep neural network to make a decision about which direction to move every frame.

Check out the following website for a more detailed description of the project and demos:  
https://craighaber.github.io/AI-for-Snake-Game/

There are 3 programs to check out in this project:
	
	playSnakeGame.py
		You can try the classic snake game here by using the arrow keys to move the snake
	testTrainedAgents.py
		Observe some of the best Snake Game agents trained with the genetic algorithm!
	Experiments.py
		To train Agents for all 3 algorithms (A2C, DQN and PPO) and 9 Reward Structures

Dependecies:

   1. Python version of 3.7 or higher.
   2. The Python library pygame.
        Type "pip install pygame" in the command prompt or terminal to install it.
        For more instructions on installing pygame, you can use the following link:
        https://www.pygame.org/wiki/GettingStarted 

Instructions:

	playSnakeGame.py

		Use the arrow keys to move up, down, left, or right.
		The goal is to get the snake as long as possible by eating fruit (the red squares)
		You will die and then automatically restart the game if:
			1. The snake hits a wall.
			2. The snake hits its own body.

	testTrainedAgents.py

		Follow the menu prompts in the command prompt/terminal to select which snake 
		you would like to observe, and then watch as the agent plays the game
		in a new window.
