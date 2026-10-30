"""Tabular Q-learning agent for a grid maze."""
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(42); np.random.seed(42)
maze = np.array([[0,0,0,0,0],[0,1,1,1,0],[0,0,0,1,0],[0,1,0,0,0],[0,0,0,1,0]])
start=(0,0); goal=(4,4); actions=[(-1,0),(1,0),(0,-1),(0,1)]
q=np.zeros((5,5,4)); rewards=[]
def step(pos, action):
 nr,nc=pos[0]+actions[action][0],pos[1]+actions[action][1]
 if not (0<=nr<5 and 0<=nc<5) or maze[nr,nc]==1:return pos,-5,False
 nxt=(nr,nc);return nxt,(100 if nxt==goal else -1),nxt==goal
for episode in range(4000):
 pos=start;total=0;epsilon=max(0.05,1.0-episode/3000)
 for _ in range(120):
  a=random.randrange(4) if random.random()<epsilon else int(np.argmax(q[pos]))
  nxt,r,done=step(pos,a)
  q[pos][a]+=0.2*(r+0.95*np.max(q[nxt])*(not done)-q[pos][a]);total+=r;pos=nxt
  if done:break
 rewards.append(total)
np.save('maze_q_table.npy',q)
path=[start];pos=start
for _ in range(40):
 a=int(np.argmax(q[pos]));nxt,_,done=step(pos,a)
 if nxt==pos or nxt in path:break
 path.append(nxt);pos=nxt
 if done:break
print('Learned path:',path,'Reached goal:',pos==goal)
plt.plot(np.convolve(rewards,np.ones(100)/100,mode='valid'));plt.title('Q-Learning Moving Average Reward');plt.xlabel('Episode');plt.ylabel('Reward');plt.tight_layout();plt.savefig('maze_learning_curve.png');plt.close()
plt.imshow(maze,cmap='Greys');plt.plot([p[1] for p in path],[p[0] for p in path],'-o');plt.xticks(range(5));plt.yticks(range(5));plt.title('Learned Maze Path');plt.savefig('maze_path.png')
