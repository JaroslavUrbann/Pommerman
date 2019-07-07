#include "bboard.hpp"
#include "agents.hpp"
#include "strategy.hpp"
#include "step_utility.hpp"
#include <chrono>

using namespace bboard;
using namespace bboard::strategy;

namespace agents
{
BerlinAgent::BerlinAgent()
{
}


bool BerlinAgent::_CheckPos2(const State* state, bboard::Position pos)
{
    return !util::IsOutOfBounds(pos) && IS_WALKABLE_OR_AGENT(state->board[pos.y][pos.x]);
}

float BerlinAgent::laterBetter(float reward, int timestaps)
{
    if(reward == 0.0f)
        return reward;

    if(reward > 0)
        return reward * (1.0f / (float)std::pow(0.98f, timestaps));
    else
        return reward * (float)std::pow(0.98f, timestaps);
}

float BerlinAgent::soonerBetter(float reward, int timestaps)
{
    if(reward == 0.0f)
        return reward;

    if(reward < 0)
        return reward * (1.0f / (float)std::pow(0.98f, timestaps));
    else
        return reward * (float)std::pow(0.98f, timestaps);
}

float BerlinAgent::scoreState(State * state) {
    float point = laterBetter(-10 * state->agents[state->ourId].dead, state->agents[state->ourId].diedAt - state->timeStep);
    point += laterBetter(-7 * state->agents[state->teammateId].dead, state->agents[state->teammateId].diedAt - state->timeStep);
    point += 3 * soonerBetter(state->agents[state->enemy1Id].dead, state->agents[state->enemy1Id].diedAt - state->timeStep);
    point += 3 * soonerBetter(state->agents[state->enemy2Id].dead, state->agents[state->enemy2Id].diedAt - state->timeStep);
    point += 0.3f * state->agents[state->ourId].woodDemolished;
    point += 0.5f * (state->agents[state->ourId].extraBombPowerupPoints + state->agents[state->ourId].firstKickPowerupPoints + state->agents[state->ourId].otherKickPowerupPoints + state->agents[state->ourId].extraRangePowerupPoints);

    if(state->agents[state->enemy1Id].x >= 0)
        point -= (std::abs(state->agents[state->enemy1Id].x - state->agents[state->ourId].x) + std::abs(state->agents[state->enemy1Id].y - state->agents[state->ourId].y)) / 100.0f;
    if(state->agents[state->enemy2Id].x >= 0)
        point -= (std::abs(state->agents[state->enemy2Id].x - state->agents[state->ourId].x) + std::abs(state->agents[state->enemy2Id].y - state->agents[state->ourId].y)) / 100.0f;

    if(moves_in_chain[0] != 0) point += 0.001f; //not IDLE
    if(lastMoveWasBlocked && moves_in_chain[0] == lastBlockedMove)
        point -= 0.1f;

    return point;
}

float BerlinAgent::runAlreadyPlantedBombs(State * state)
{
    for(int i=0; i<10; i++)//TODO: explode-all-bombs?
    {
        //bboard::Step(newstate4, moves);
        util::TickBombs(*state);
        state->relTimeStep++;
        //simulatedSteps++;
    }

    float point = scoreState(state);
    /*if(point > bestPoint)
    {
        bestPoint = point;
        best_moves_in_chain = moves_in_chain;
    }*/

    return point;
}

float BerlinAgent::runOneStep(const bboard::State * state, int depth)
{
    bboard::Move moves_in_one_step[4]; //was: moves
    moves_in_one_step[0] = bboard::Move::IDLE;
    moves_in_one_step[1] = bboard::Move::IDLE;
    moves_in_one_step[2] = bboard::Move::IDLE;
    moves_in_one_step[3] = bboard::Move::IDLE;

    const AgentInfo& a = state->agents[state->ourId];
    float maxPoint = -100;
    for(int move=0; move<6; move++)
    {
        // if we don't have bomb
        if(move == (int)bboard::Move::BOMB && a.maxBombCount - a.bombCount <= 0)
            continue;
        // if bomb is already under us
        if(depth == 0 && move == (int)bboard::Move::BOMB && state->HasBomb(a.x, a.y))
            continue;
        // if move is impossible
        if(move>0 && move<5 && !_CheckPos2(state, bboard::util::DesiredPosition(a.x, a.y, (bboard::Move)move)))
            continue;

        moves_in_one_step[state->ourId] = (bboard::Move) move;
        moves_in_chain.AddElem(move);

        float maxTeammate = -100;
        for(int moveT=0; moveT<6; moveT++) {
            if (moveT > 0) {
                if (depth > 0 || (state->agents[state->teammateId].dead || state->agents[state->teammateId].x < 0)) break;
                // if move is impossible
                if (moveT > 0 && moveT < 5 && !_CheckPos2(state, bboard::util::DesiredPosition(state->agents[state->teammateId].x, state->agents[state->teammateId].y, (bboard::Move) moveT)))
                    continue;
                if (moveT == (int) bboard::Move::BOMB && state->agents[state->teammateId].maxBombCount - state->agents[state->teammateId].bombCount <= 0)
                    continue;
            }

            moves_in_one_step[state->teammateId] = (bboard::Move) moveT;

            float minPointE1 = 100;
            for (int moveE1 = 0; moveE1 < 6; moveE1++) {
                if (moveE1 > 0) {
                    if (depth > 0 || (state->agents[state->enemy1Id].dead || state->agents[state->enemy1Id].x < 0)) break;
                    // if move is impossible
                    if (moveE1 > 0 && moveE1 < 5 && !_CheckPos2(state, bboard::util::DesiredPosition(state->agents[state->enemy1Id].x, state->agents[state->enemy1Id].y, (bboard::Move) moveE1)))
                        continue;
                    if (moveE1 == (int) bboard::Move::BOMB &&
                        state->agents[state->enemy1Id].maxBombCount - state->agents[state->enemy1Id].bombCount <= 0)
                        continue;
                }

                moves_in_one_step[state->enemy1Id] = (bboard::Move) moveE1;

                float minPointE2 = 100;
                for (int moveE2 = 0; moveE2 < 6; moveE2++) {
                    if (moveE2 > 0) {
                        if (depth > 0 || (state->agents[state->enemy2Id].dead || state->agents[state->enemy2Id].x < 0)) break;
                        // if move is impossible
                        if (moveE2 > 0 && moveE2 < 5 && !_CheckPos2(state, bboard::util::DesiredPosition(state->agents[state->enemy2Id].x, state->agents[state->enemy2Id].y, (bboard::Move) moveE2)))
                            continue;
                        if (moveE2 == (int) bboard::Move::BOMB &&
                            state->agents[state->enemy2Id].maxBombCount - state->agents[state->enemy2Id].bombCount <= 0)
                            continue;
                    }

                    moves_in_one_step[state->enemy2Id] = (bboard::Move) moveE2;

                    bboard::State *newstate = new bboard::State(*state);
                    newstate->relTimeStep++;

                    bboard::Step(newstate, moves_in_one_step);
                    simulatedSteps++;

                    float point;
                    if (depth < 4)
                    {
#ifdef TIME_LIMIT_ON
                    size_t millis = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time).count();
                    if (millis < 90 || (depth == 0 && millis < 95))
                        point = runOneStep(newstate, depth + 1);
                    else
                        point = runAlreadyPlantedBombs(newstate);
#else
                        point = runOneStep(newstate, depth + 1);
#endif
                        }
                        else
                        point = runAlreadyPlantedBombs(newstate);

                    if (point < minPointE2) { minPointE2 = point; }

                    delete newstate;
                }
                if (minPointE2 < minPointE1) { minPointE1 = minPointE2; }
            }
            if (minPointE1 > maxTeammate) { maxTeammate = minPointE1;}
        }

        moves_in_chain.RemoveAt(moves_in_chain.count - 1);
        best_moves_in_chain.count = std::max(depth+1, best_moves_in_chain.count);
        if (maxTeammate > maxPoint) { maxPoint = maxTeammate; best_moves_in_chain[depth] = move;}
    }

    return maxPoint;
}

Move BerlinAgent::act(const State* state)
{
    simulatedSteps = 0;
    bestPoint = -100.0f;
    const AgentInfo& a = state->agents[state->ourId];
    if(state->timeStep > 1 && (expectedPosInNewTurn.x != a.x || expectedPosInNewTurn.y != a.y))
    {
        //std::cout << "Couldn't move to " << expectedPosInNewTurn.y << ":" << expectedPosInNewTurn.x << std::endl;
        lastMoveWasBlocked = true;
        lastBlockedMove = best_moves_in_chain[0];
    }else{
        lastMoveWasBlocked = false;
    }

    float point = runOneStep(state, 0);

    /*std::cout << "turn#" << state->timeStep << " ourId:" << state->ourId << " point: " << point << " selected: ";
    for(int i=0; i<best_moves_in_chain.count; i++)
        std::cout << (int)best_moves_in_chain[i] << " > ";
    std::cout << " simulated steps: " << simulatedSteps << std::endl;*/

    totalSimulatedSteps += simulatedSteps;
    turns++;
    expectedPosInNewTurn = bboard::util::DesiredPosition(a.x, a.y, (bboard::Move)best_moves_in_chain[0]);
    return (bboard::Move)best_moves_in_chain[0];
}

void BerlinAgent::PrintDetailedInfo()
{
}

}
