#ifndef STEP_UTILITY_H
#define STEP_UTILITY_H

#include "bboard.hpp"

namespace bboard::util
{

/**
 * @brief DesiredPosition returns the x and y values of the agents
 * destination
 * @param x Current agents x position
 * @param y Current agents y position
 * @param m The desired move
 */
Position DesiredPosition(int x, int y, Move m);


/**
 * @brief FillDestPos Fills an array of destination positions.
 * @param s The State
 * @param m An array of all agent moves
 * @param p The array to be filled wih dest positions
 */
void FillDestPos(State* s, Move m[AGENT_COUNT], Position p[AGENT_COUNT]);

/**
 * @brief FixSwitchMove Fixes the desired positions if the agents want
 * switch places in one step.
 * @param s The state
 * @param desiredPositions an array of desired positions
 */
bool FixSwitchMove(State* s, Position desiredPositions[AGENT_COUNT]);

void MoveBombs(State* state, Position d[AGENT_COUNT]);

/**
 * TODO: Fill doc for dependency resolving
 *
 */
int ResolveDependencies(State* s, Position des[AGENT_COUNT],
                        int dependency[AGENT_COUNT], int chain[AGENT_COUNT]);

/**
 * @brief TickFlames Counts down all flames in the flame queue
 * (and possible extinguishes the flame)
 */
void TickFlames(State& state);

/**
 * @brief TickBombs Counts down all bomb timers and explodes them
 * if they arrive at 10
 */
void TickBombs(State& state);
inline void TickAndMoveBombs(State& state){
    //explode timed-out bombs
    for(int i = 0; i < state.bombs.count; i++)
    {
        ReduceBombTimer(state.bombs[i]);

        if(BMB_TIME(state.bombs[i]) == 0)
        {
            __glibcxx_assert(i == 0);
            state.ExplodeTopBomb();
            i--;
        }
        else
        {
            if(BMB_VEL(state.bombs[i]) > 0) {
                Position desiredPos = bboard::util::DesiredPosition(BMB_POS_X(state.bombs[i]), BMB_POS_Y(state.bombs[i]), (bboard::Move) BMB_VEL(state.bombs[i]));
                if (_CheckPos_any(&state, desiredPos.x, desiredPos.y)) {
                    state.board[BMB_POS_Y(state.bombs[i])][BMB_POS_X(state.bombs[i])] = PASSAGE;
                    SetBombPosition(state.bombs[i], desiredPos.x, desiredPos.y);
                    bool explodes = IS_FLAME(state.board[desiredPos.y][desiredPos.x]);
                    state.board[desiredPos.y][desiredPos.x] = BOMB;

                    if (explodes) {
                        state.ExplodeBomb(i);
                        i--;
                    }
                }else{
                    SetBombVelocity(state.bombs[i], 0);
                }
            }
        }
    }
}
void TickAndMoveBombs10(State& state);

/**
 * @brief ConsumePowerup Lets an agent consume a powerup
 * @param agentID The agent's ID that consumes the item
 * @param powerUp A powerup item. If it's something else,
 * this function will do nothing.
 */
void ConsumePowerup(State& state, int agentID, int powerUp);

/**
 * @brief PrintDependency Prints a dependency array in a nice
 * format (stdout)
 * @param dependency Integer array that contains the dependencies
 */
void PrintDependency(int dependency[AGENT_COUNT]);

/**
 * @brief PrintDependencyChain Prints a dependency chain in a nice
 * format (stdout)
 * @param dependency Integer array that contains the dependencies
 * @param chain Integer array that contains all chain roots.
 * (-1 is a skip)
 */
void PrintDependencyChain(int dependency[AGENT_COUNT], int chain[AGENT_COUNT]);

/**
 * @brief HasDPCollision Checks if the given agent has a destination
 * position collision with another agent
 * @param The agent that's checked for collisions
 * @return True if there is at least one collision
 */
bool HasDPCollision(const State& state, Position dp[AGENT_COUNT], int agentID);

/**
 * @brief IsOutOfBounds Checks wether a given position is out of bounds
 */
inline bool IsOutOfBounds(const Position& pos)
{
    return pos.x < 0 || pos.y < 0 || pos.x >= BOARD_SIZE || pos.y >= BOARD_SIZE;
}

/**
 * @brief IsOutOfBounds Checks wether a given position is out of bounds
 */
inline bool IsOutOfBounds(const int& x, const int& y)
{
    return x < 0 || y < 0 || x >= BOARD_SIZE || y >= BOARD_SIZE;
}

inline bool AreOppositeMoves(int m1, int m2)
{
    return m1 > 0 && m2 > 0 && m1 < 5 && m2 < 5 && ((m1 + m2) == 3 || (m1 + m2) == 7);
}

}

#endif // STEP_UTILITY_H
