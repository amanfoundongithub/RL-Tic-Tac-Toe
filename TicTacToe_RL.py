import numpy
import pickle
from tqdm import tqdm

N_ROWS = 3
N_COLS = 3

class State: 
    def __init__(self, player_1, player_2):
        # Board configuration
        self.board = numpy.zeros((N_ROWS, N_COLS))

        # Player 1 and Player 2
        self.player_1 = player_1 
        self.player_2 = player_2 

        # Move : True if Player 1 turn False if Player 2 turn
        self.move = True 

        # End game signal
        self.end = False 
        pass 

    def reset(self):
        # Board configuration
        self.board = numpy.zeros((N_ROWS, N_COLS))

        # Move : True if Player 1 turn False if Player 2 turn
        self.move = True 

        # End game signal
        self.end = False 
        pass 


    def updateState(self, position):
        move = self.move
        if move == True:
            symbol = 1
        else: 
            symbol = -1
        self.board[position] = symbol 
        self.move = ~ self.move 
        return 
    

    def availablePositions(self):
        lst = []
        for i in range(N_ROWS):
            for j in range(N_COLS):
                if self.board[i][j] == 0:
                    lst.append((i,j))
        
        return lst 
    
    def winner(self):
        for i in range(N_ROWS):
            if sum(self.board[i]) == 3:
                self.end = True
                return 1
            elif sum(self.board[i]) == -3:
                self.end = True
                return -1
        
        for i in range(N_COLS):
            if sum(self.board[:,i]) == 3:
                self.end = True
                return 1
            elif sum(self.board[:,i]) == -3:
                self.end = True
                return -1
        
        s = 0 
        for i in range(N_COLS):
            s += self.board[i,i] 
        if s == 3:
            self.end = True
            return 1 
        elif s == -3:
            self.end = True
            return -1 
        
        s = 0
        for i in range(N_COLS):
            s += self.board[i][N_COLS - i - 1]
        if s == 3:
            self.end = True
            return 1 
        elif s == -3:
            self.end = True
            return -1
        
        available = self.availablePositions()
        if len(available) == 0:
            self.end = True 
            return 0
        
        self.end = False 
        return None 
    
    def reward(self):
        result = self.winner()

        if result == 0 or result == None: 
            self.player_1.rewardFeedback(0.1) 
            self.player_2.rewardFeedback(0.9)
            pass 
        elif result == 1:
            self.player_1.rewardFeedback(1) 
            self.player_2.rewardFeedback(0)
            pass 
        elif result == -1:
            self.player_1.rewardFeedback(0) 
            self.player_2.rewardFeedback(1)
            pass 
    
    def showBoard(self):
        for i in range(0, N_ROWS):
            print('-------------')
            out = '| '
            for j in range(0, N_COLS):
                if self.board[i, j] == 1:
                    token = 'x'
                if self.board[i, j] == -1:
                    token = 'o'
                if self.board[i, j] == 0:
                    token = ' '
                out += token + ' | '
            print(out)
        print('-------------')
    
    def play(self, no_of_games = 1000):
        for i in tqdm(range(no_of_games)):
            #if i % 1000 == 0:
                #print("Round #" + str(i) + " of training done...")

            while not self.end: 
                # Player 1
                positions = self.availablePositions() 
                p1_action = self.player_1.chooseAction(positions, self.board, 1)
                self.updateState(p1_action)

                board_hash = str(self.board.reshape((N_ROWS * N_COLS,)))

                self.player_1.addState(board_hash) 

                win = self.winner()

                if win is not None:
                    
                    self.reward()
                    self.player_1.reset()
                    self.player_2.reset()
                    self.reset()
                    break 
                
                else:
                    positions = self.availablePositions() 
                    p2_action = self.player_2.chooseAction(positions, self.board, -1)
                    self.updateState(p2_action)

                    board_hash = str(self.board.reshape((N_ROWS * N_COLS,)))

                    self.player_2.addState(board_hash) 

                    win = self.winner()

                    if win is not None:
                        self.reward()
                        self.player_1.reset()
                        self.player_2.reset()
                        self.reset()
                        break 
            
            win = self.winner()

        self.player_1.savePolicy()
    def play2(self):
        while not self.end:
            # Player 1
            positions = self.availablePositions()
            p1_action = self.player_1.chooseAction(positions, self.board, 1)
            # take action and upate board state
            self.updateState(p1_action)
            print(self.player_1.name + "'s Move:") 
            self.showBoard()
            
            # check board status if it is end
            win = self.winner()
    
            if win is not None:
                if win == 1:
                    print(self.player_1.name, "wins!")
                else:
                    print("The match is a Tie!!!")
                self.reset()
                break

            else:
                # Player 2
                positions = self.availablePositions()
                p2_action = self.player_2.chooseAction(positions)
                self.updateState(p2_action)
                print(self.player_2.name + "'s Move:")
                self.showBoard()
                win = self.winner()
                if win is not None:
                    if win == -1:
                        print(self.player_2.name, "wins!")
                    else:
                        print("The match is a Tie!!!")
                    self.reset()
                    break

class Agent:
    
    def __init__(self, name, exp_rate = 0.3):
        self.name = name 
        self.states = []
        self.learning_rate = 0.01
        self.gamma = 0.5
        self.states_values = dict()
        
        # epsilon greedy method 
        self.exp_rate = exp_rate
    
    def chooseAction(self, positions, current_board, symbol):

        random_action = numpy.random.uniform(0,1) 

        if random_action <= self.exp_rate:
            idx = numpy.random.choice(len(positions))
            action = positions[idx] 
        
        else: 
            value_max = -1e4
            for p in positions:
                n_board = current_board.copy()
                n_board[p] = symbol 
                n_board_hash = str(n_board.reshape((N_ROWS * N_COLS,)))

                value = 0
                if self.states_values.get(n_board_hash) is not None: 
                    value = self.states_values.get(n_board_hash)
                
                if value > value_max:
                    value_max = value 
                    action = p 
        
        return action 

    def addState(self, state):
        self.states.append(state) 
        return 

    def rewardFeedback(self, reward):
        for st in reversed(self.states):
            if self.states_values.get(st) is None: 
                self.states_values[st] = 0 
                
            self.states_values[st] += self.learning_rate * (self.gamma * reward - self.states_values[st])
            reward = self.states_values[st] 
    
    def reset(self):
        self.states = [] 
        
    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_values, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_values = pickle.load(fr)
        fr.close()

class Player:
    def __init__(self, name):
        self.name = name 
    
    def chooseAction(self, positions):
        action = None
        while action not in positions:
            print("USER ACTION...")
            row = int(input("Enter row number: "))
            col = int(input("Enter col number: "))

            action = (row - 1,col - 1)
        return action   
            
    def addState(self, state):
        pass 
    def rewardFeedback(self, reward):
        pass

if __name__ == "__main__":
    # training
    p1 = Agent("Adam")
    p2 = Agent("Eve")

    st = State(p1, p2)
    print("Training phase begins...")
    st.play(200000)
    print("...Training phase ends")
    print()
    print()
    print("---------------------------------------------------------------------------------------------------------------------")
    # play with human
    p1 = Agent("Jeff", exp_rate=0)
    p1.loadPolicy('policy_Adam')
    print("Hi, My Name is Jeff. Let's play Tic Tac Toe!")

    while True:
        name = input("Enter your name: ")
        p2 = Player(name)
        st = State(p1, p2)
        st.play2()
        y = input("Want to play again?")
        if y == 'no':
            exit(0)