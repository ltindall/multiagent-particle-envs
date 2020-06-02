import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2 # communication dimension set but communication info never observed

        # agent properties 
        num_trapped_agents = 0
        num_rescue_agents = 3
        num_adversary_agents = 0
        num_agents = num_adversary_agents + num_rescue_agents + num_trapped_agents
        world.agents = []
        
        # add trapped agents
        for i in range(num_trapped_agents): 
            agent = Agent()
            agent.name = 'trapped_agent_%d' % i
            agent.collide = True
            agent.silent = True
            agent.agent_type = 'trapped'
            agent.adversary = False
            #agent.trapped = True
            agent.size =  0.05
            agent.accel =  4.0
            agent.max_speed = 1.3
            agent.color = np.array([0.35, 0.85, 0.35]) # green 
            world.agents.append(agent)
        
        # add rescue agents 
        for i in range(num_rescue_agents): 
            agent = Agent()
            agent.name = 'rescue_agent_%d' % i
            agent.collide = True
            agent.silent = True
            agent.agent_type = 'rescue'
            agent.adversary = False
            #agent.trapped = False 
            agent.size =  0.05
            agent.accel =  4.0
            agent.max_speed = 1.3
            agent.color = np.array([0.35, 0.35, 0.75]) # blue
            world.agents.append(agent)

        # add adversary agents 
        for i in range(num_adversary_agents): 
            agent = Agent()
            agent.name = 'adversary_agent_%d' % i
            agent.collide = True
            agent.silent = True
            agent.agent_type = 'adversary'
            agent.adversary = True
            agent.size =  0.075
            agent.accel =  3.0
            agent.max_speed = 1.0
            agent.color = np.array([0.85, 0.35, 0.35]) # red 
            world.agents.append(agent)

        # landmark properties 
        num_goal_landmark = 1
        num_move_landmark = 1 
        num_static_landmark = 4
        num_landmarks = num_goal_landmark + num_move_landmark + num_static_landmark
        world.landmarks = []

        for i in range(num_goal_landmark): 
            landmark = Landmark()
            landmark.name = 'goal_landmark_%d' % i
            landmark.land_type = 'goal'
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
            landmark.color = np.array([0.25, 0.85, 0.25]) # green
            world.landmarks.append(landmark)
        
        for i in range(num_move_landmark): 
            landmark = Landmark()
            landmark.name = 'move_landmark_%d' % i
            landmark.land_type = 'move'
            landmark.collide = True
            landmark.movable = True
            landmark.size = 0.15
            landmark.boundary = False
            landmark.color = np.array([0.25, 0.25, 0.95]) # blue
            world.landmarks.append(landmark)
        
        for i in range(num_static_landmark): 
            landmark = Landmark()
            landmark.name = 'static_landmark_%d' % i
            landmark.land_type = 'static'
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.1
            landmark.boundary = False
            landmark.color = np.array([0.25, 0.25, 0.25]) # grey
            world.landmarks.append(landmark)
        
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.agent_type == 'adversary':
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def trapped_agents(self, world):
        return [agent for agent in world.agents if agent.agent_type == 'trapped']

    # return all agents that are not adversaries
    def rescue_agents(self, world):
        return [agent for agent in world.agents if agent.agent_type == 'rescue']

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.agent_type == 'adversary']

    # return all goal landmarks
    def goals(self, world): 
        return [landmark for landmark in world.landmarks if landmark.land_type == 'goal']

    # return all moveable landmarks
    def moveable(self, world): 
        return [landmark for landmark in world.landmarks if landmark.land_type == 'move']

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.agent_type == 'adversary' else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):

        moveable_landmarks = self.moveable(world)
        goal_landmarks = self.goals(world)

        reward = 0 
        for m in moveable_landmarks: 
            reward -= min([np.sqrt(np.sum(np.square(m.state.p_pos - g.state.p_pos))) for g in goal_landmarks])

        return reward

    ### NOT USED IN THIS SCENARIO 
    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = True
        trapped_agents = self.trapped_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in trapped_agents])
        if agent.collide:
            for ag in trapped_agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        #comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            #comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if other.agent_type != 'adversary':
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)
