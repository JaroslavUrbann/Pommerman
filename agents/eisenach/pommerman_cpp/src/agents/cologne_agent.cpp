#include "bboard.hpp"
#include "agents.hpp"
#include "strategy.hpp"
#include "step_utility.hpp"
#include <list>
#include <cstring>

using namespace bboard;
using namespace bboard::strategy;

namespace agents {
    CologneAgent::CologneAgent() {
        std::random_device rd;  // non explicit seed
        rng = std::mt19937_64(rd());
        intDist = std::uniform_int_distribution<int>(0, 4); // no bombs
    }

    bool CologneAgent::_CheckPos2(const State *state, bboard::Position pos) {
        return !util::IsOutOfBounds(pos) && IS_WALKABLE_OR_AGENT(state->board[pos.y][pos.x]);
    }

    bool CologneAgent::_CheckPos2(const State *state, int x, int y) {
        return !util::IsOutOfBounds(x, y) && IS_WALKABLE_OR_AGENT(state->board[y][x]);
    }

    bool CologneAgent::_CheckPos3(const State *state, int x, int y) {
        return !util::IsOutOfBounds(x, y) && state->board[y][x] != WOOD && state->board[y][x] != RIGID;
    }

    float CologneAgent::laterBetter(float reward, int timestaps) {
        if (reward == 0.0f)
            return reward;

        if (reward > 0)
            return reward * (1.0f / (float) std::pow(reward_sooner_later_ratio, timestaps));
        else
            return reward * (float) std::pow(reward_sooner_later_ratio, timestaps);
    }

    float CologneAgent::soonerBetter(float reward, int timestaps) {
        if (reward == 0.0f)
            return reward;

        if (reward < 0)
            return reward * (1.0f / (float) std::pow(reward_sooner_later_ratio, timestaps));
        else
            return reward * (float) std::pow(reward_sooner_later_ratio, timestaps);
    }

    StepResult CologneAgent::scoreState(State *state) {
        StepResult stepRes;
        float teamBalance = (ourId < 2 ? 1.01f : 0.99f);
        float point = 0.0f;
        if (state->agents[ourId].dead) {
            point += laterBetter(-10 * state->agents[ourId].dead, state->agents[ourId].diedAt - state->timeStep);
#ifdef GM_DEBUGMODE_COMMENTS
            stepRes.comment += "I_die ";
#endif
        }
        if (state->agents[teammateId].x >= 0 && state->agents[teammateId].dead) {
            point += laterBetter(-10 * state->agents[teammateId].dead,
                                 state->agents[teammateId].diedAt - state->timeStep);
#ifdef GM_DEBUGMODE_COMMENTS
            stepRes.comment += "kanka_dies ";
#endif
        }
        if (state->agents[enemy1Id].x >= 0 && state->agents[enemy1Id].dead) {
            point += 3 * soonerBetter(state->agents[enemy1Id].dead, state->agents[enemy1Id].diedAt - state->timeStep);
#ifdef GM_DEBUGMODE_COMMENTS
            stepRes.comment += "sapsaltavuk1_dies ";
#endif
        }
        if (state->agents[enemy2Id].x >= 0 && state->agents[enemy2Id].dead) {
            point += 3 * soonerBetter(state->agents[enemy2Id].dead, state->agents[enemy2Id].diedAt - state->timeStep);
#ifdef GM_DEBUGMODE_COMMENTS
            stepRes.comment += "sapsaltavuk2_dies ";
#endif
        }
        point += 0.3f * state->agents[ourId].woodDemolished;
        point += 0.3f * state->agents[teammateId].woodDemolished;
        point -= 0.3f * state->agents[enemy1Id].woodDemolished;
        point -= 0.3f * state->agents[enemy2Id].woodDemolished;

        point += reward_collectedPowerup * (state->agents[state->ourId].extraBombPowerupPoints + state->agents[state->ourId].firstKickPowerupPoints + state->agents[state->ourId].otherKickPowerupPoints + state->agents[state->ourId].extraRangePowerupPoints) * teamBalance;
        point += reward_collectedPowerup * (state->agents[state->teammateId].extraBombPowerupPoints + state->agents[state->teammateId].firstKickPowerupPoints + state->agents[state->teammateId].otherKickPowerupPoints + state->agents[state->teammateId].extraRangePowerupPoints) / teamBalance;
        point -= reward_collectedPowerup * (state->agents[state->enemy1Id].extraBombPowerupPoints + state->agents[state->enemy1Id].firstKickPowerupPoints + state->agents[state->enemy1Id].otherKickPowerupPoints + state->agents[state->enemy1Id].extraRangePowerupPoints);
        point -= reward_collectedPowerup * (state->agents[state->enemy2Id].extraBombPowerupPoints + state->agents[state->enemy2Id].firstKickPowerupPoints + state->agents[state->enemy2Id].otherKickPowerupPoints + state->agents[state->enemy2Id].extraRangePowerupPoints);

        if (state->aliveAgents == 0) {
            //point += soonerBetter(??, state->relTimeStep); //we win
#ifdef GM_DEBUGMODE_COMMENTS
            stepRes.comment += "tie ";
#endif
        } else if (state->aliveAgents < 3) {
            if (state->agents[ourId].dead && state->agents[teammateId].dead) {
                point += laterBetter(-20.0f, state->relTimeStep); //we lost
#ifdef GM_DEBUGMODE_COMMENTS
                stepRes.comment += "we_lost ";
#endif
            }
            if (state->agents[enemy1Id].dead && state->agents[enemy2Id].dead) {
                point += soonerBetter(+20.0f, state->relTimeStep); //we win
#ifdef GM_DEBUGMODE_COMMENTS
                stepRes.comment += "we_win ";
#endif
            }
        }

        if (state->agents[enemy1Id].x >= 0)
            point -= (std::abs(state->agents[enemy1Id].x - state->agents[ourId].x) +
                      std::abs(state->agents[enemy1Id].y - state->agents[ourId].y)) / reward_move_to_enemy;
        if (state->agents[enemy2Id].x >= 0)
            point -= (std::abs(state->agents[enemy2Id].x - state->agents[ourId].x) +
                      std::abs(state->agents[enemy2Id].y - state->agents[ourId].y)) / reward_move_to_enemy;

        for (int i = 0; i < state->powerup_kick.count; i++)
            point -= (std::abs(state->powerup_kick[i].x - state->agents[ourId].x) +
                      std::abs(state->powerup_kick[i].y - state->agents[ourId].y)) / reward_move_to_pickup;
        for (int i = 0; i < state->powerup_incr.count; i++)
            point -= (std::abs(state->powerup_incr[i].x - state->agents[ourId].x) +
                      std::abs(state->powerup_incr[i].y - state->agents[ourId].y)) / reward_move_to_pickup;
        for (int i = 0; i < state->powerup_extrabomb.count; i++)
            point -= (std::abs(state->powerup_extrabomb[i].x - state->agents[ourId].x) +
                      std::abs(state->powerup_extrabomb[i].y - state->agents[ourId].y)) / reward_move_to_pickup;
        //Following woods decrease points a little bit, I tried 3 different test setups. It would help, but it doesnt. Turned off.
        if (seenAgents == 0) {
            for (int i = 0; i < state->woods.count; i++)
                point -= (std::abs(state->woods[i].x - state->agents[ourId].x) +
                          std::abs(state->woods[i].y - state->agents[ourId].y)) / 1000.0f;
        }

        if (moves_in_chain[0] == 0) point -= reward_first_step_idle;
        if (lastMoveWasBlocked && ((state->timeStep / 4 + ourId) % 4) == 0 && moves_in_chain[0] == lastBlockedMove)
            point -= 0.1f;
        if (sameAs6_12_turns_ago && ((state->timeStep / 4 + ourId) % 4) == 0 && moves_in_chain[0] == moveHistory[moveHistory.count - 6])
            point -= 0.1f;

        if(!state->agents[teammateId].dead && state->agents[teammateId].x >= 0 && leadsToDeadEnd[state->agents[teammateId].x + BOARD_SIZE * state->agents[teammateId].y])
            point -= 0.0004f;
        if(!state->agents[enemy1Id].dead && state->agents[enemy1Id].x >= 0 && leadsToDeadEnd[state->agents[enemy1Id].x + BOARD_SIZE * state->agents[enemy1Id].y])
            point += 0.0004f;
        if(!state->agents[enemy2Id].dead && state->agents[enemy2Id].x >= 0 && leadsToDeadEnd[state->agents[enemy2Id].x + BOARD_SIZE * state->agents[enemy2Id].y])
            point += 0.0004f;
        for (int i = 0; i < positions_in_chain.count; i++) {
            if (leadsToDeadEnd[positions_in_chain[i].x + BOARD_SIZE * positions_in_chain[i].y])
                point -= 0.001f;
            break; //only for the first move, as the leadsToDeadEnd can be deprecated if calculated with flames, bombs, woods. Yields better results.
        }

#ifdef GM_DEBUGMODE_ON
        stepRes.point = point;
#else
        stepRes = point;
#endif
        return stepRes;
    }

    StepResult CologneAgent::runAlreadyPlantedBombs(State *state) {
        for (int i = 0; i < 10; i++) {
            //Exit if match decided, maybe we would die later from an other bomb, so that disturbs pointing and decision making
            if (state->aliveAgents < 2 || (state->aliveAgents == 2 &&
                                           ((state->agents[0].dead && state->agents[2].dead) ||
                                            (state->agents[1].dead && state->agents[3].dead))))
                break;
            //bboard::Step(state, moves);
            util::TickBombs(*state);
            state->relTimeStep++;
            //simulatedSteps++;
        }

        return scoreState(state);
    }

//#define RANDOM_TIEBREAK //With nobomb-random-tiebreak: 10% less simsteps, 3% less wins :( , 5-10% less ties against simple. Turned off by default. See log_test_02_tie.txt
//#define SCENE_HASH_MEMORY //8-10x less simsteps, but 40% less wins :((
    StepResult CologneAgent::runOneStep(const bboard::State *state, const int depth) {
        StepResult stepRes;
        bboard::Move moves_in_one_step[4];
        const AgentInfo &a = state->agents[ourId];
#ifdef GM_DEBUGMODE_ON
        stepRes.point = -100;
#else
        stepRes = -100;
#endif

#ifdef RANDOM_TIEBREAK
        FixedQueue<int, 6> bestmoves;
#endif
        //int moves[]{1,2,3,4,0,5};
        //for(int move : moves)
        for (int move = 0; move < 6; move++)
        {
            Position desiredPos = bboard::util::DesiredPosition(a.x, a.y, (bboard::Move) move);
            // if we don't have bomb
            if (move == (int) bboard::Move::BOMB && a.maxBombCount - a.bombCount <= 0)
                continue;
            // if bomb is already under us
            if (move == (int) bboard::Move::BOMB && state->HasBomb(a.x, a.y))
                continue;
            // if move is impossible
            if (move > 0 && move < 5 && !_CheckPos2(state, desiredPos))
                continue;
            //no two opposite steps please!
            if (depth > 0 && move > 0 && move < 5 && moves_in_chain[depth - 1] > 0 && moves_in_chain[depth - 1] < 5 &&
                std::abs(moves_in_chain[depth - 1] - move) == 2)
                continue;

            moves_in_one_step[ourId] = (bboard::Move) move;
            moves_in_chain.AddElem(move);

            float maxTeammate = -100;
            StepResult futureStepsT;
            for (int moveT = 5; moveT >= 0; moveT--) {
                if (moveT > 0) {
                    if (depth >= teammateIteration ||
                        (state->agents[teammateId].dead || state->agents[teammateId].x < 0))
                        continue;
                    // if move is impossible
                    if (moveT > 0 && moveT < 5 && !_CheckPos2(state,
                                                              bboard::util::DesiredPosition(state->agents[teammateId].x,
                                                                                            state->agents[teammateId].y,
                                                                                            (bboard::Move) moveT)))
                        continue;
                    if (moveT == (int) bboard::Move::BOMB &&
                        state->agents[teammateId].maxBombCount - state->agents[teammateId].bombCount <= 0)
                        continue;
                    // if bomb is already under it
                    if (moveT == (int) bboard::Move::BOMB &&
                        state->HasBomb(state->agents[teammateId].x, state->agents[teammateId].y))
                        continue;
                } else {
                    //We'll have same results with IDLE, IDLE
                    if (maxTeammate > -100 && state->agents[teammateId].x == desiredPos.x &&
                        state->agents[teammateId].y == desiredPos.y)
                        continue;
                }

                moves_in_one_step[teammateId] = (bboard::Move) moveT;

                float minPointE1 = 100;
                StepResult futureStepsE1;
                for (int moveE1 = 5; moveE1 >= 0; moveE1--) {
                    if (moveE1 > 0) {
                        if (depth >= enemyIteration1 ||
                            (state->agents[enemy1Id].dead || state->agents[enemy1Id].x < 0))
                            continue;
                        // if move is impossible
                        if (moveE1 > 0 && moveE1 < 5 && !_CheckPos2(state, bboard::util::DesiredPosition(
                                state->agents[enemy1Id].x, state->agents[enemy1Id].y, (bboard::Move) moveE1)))
                            continue;
                        if (moveE1 == (int) bboard::Move::BOMB &&
                            state->agents[enemy1Id].maxBombCount - state->agents[enemy1Id].bombCount <= 0)
                            continue;
                        // if bomb is already under it
                        if (moveE1 == (int) bboard::Move::BOMB &&
                            state->HasBomb(state->agents[enemy1Id].x, state->agents[enemy1Id].y))
                            continue;
                    } else {
                        //We'll have same results with IDLE, IDLE
                        if (minPointE1 < 100 && state->agents[enemy1Id].x == desiredPos.x &&
                            state->agents[enemy1Id].y == desiredPos.y)
                            continue;
                    }

                    moves_in_one_step[enemy1Id] = (bboard::Move) moveE1;

                    float minPointE2 = 100;
                    StepResult futureStepsE2;
                    for (int moveE2 = 5; moveE2 >= 0; moveE2--) {
                        if (moveE2 > 0) {
                            if (depth >= enemyIteration2 ||
                                (state->agents[enemy2Id].dead || state->agents[enemy2Id].x < 0))
                                continue;
                            // if move is impossible
                            if (moveE2 > 0 && moveE2 < 5 && !_CheckPos2(state, bboard::util::DesiredPosition(
                                    state->agents[enemy2Id].x, state->agents[enemy2Id].y, (bboard::Move) moveE2)))
                                continue;
                            if (moveE2 == (int) bboard::Move::BOMB &&
                                state->agents[enemy2Id].maxBombCount - state->agents[enemy2Id].bombCount <= 0)
                                continue;
                            // if bomb is already under it
                            if (moveE2 == (int) bboard::Move::BOMB &&
                                state->HasBomb(state->agents[enemy2Id].x, state->agents[enemy2Id].y))
                                continue;
                        } else {
                            //We'll have same results with IDLE, IDLE
                            if (minPointE2 < 100 && state->agents[enemy2Id].x == desiredPos.x &&
                                state->agents[enemy2Id].y == desiredPos.y)
                                continue;
                        }

                        moves_in_one_step[enemy2Id] = (bboard::Move) moveE2;

                        bboard::State *newstate = new bboard::State(*state);
                        newstate->relTimeStep++;

                        bboard::Step(newstate, moves_in_one_step);
                        simulatedSteps++;

#ifdef SCENE_HASH_MEMORY
                        uint128_t hash = ((((((((((((uint128_t)(newstate->agents[ourId].x * 11 + newstate->agents[ourId].y) * 121 +
                                         (newstate->agents[enemy1Id].dead || newstate->agents[enemy1Id].x < 0 ? 0 : newstate->agents[enemy1Id].x * 11 + newstate->agents[enemy1Id].y)) * 121 +
                                        (newstate->agents[enemy2Id].dead || newstate->agents[enemy2Id].x < 0 ? 0 : newstate->agents[enemy2Id].x * 11 + newstate->agents[enemy2Id].y)) * 121 +
                                       (newstate->agents[teammateId].dead || newstate->agents[teammateId].x < 0 ? 0 : newstate->agents[teammateId].x * 11 + newstate->agents[teammateId].y)) * 121 +
                            newstate->bombs.count)*6+
                            depth)*6+
                            (newstate->bombs.count > 0 ? newstate->bombs[newstate->bombs.count-1] : 0))*10000 +
                            (newstate->bombs.count > 1 ? newstate->bombs[newstate->bombs.count-2] : 0))*10000 +
                            (newstate->bombs.count > 2 ? newstate->bombs[newstate->bombs.count-3] : 0))*10000 +
                            (newstate->bombs.count > 3 ? newstate->bombs[newstate->bombs.count-4] : 0))*10000 +
                            newstate->agents[ourId].maxBombCount)*10 +
                            newstate->agents[ourId].bombStrength)*10 +
                            newstate->agents[0].dead*8 + newstate->agents[1].dead*4 + newstate->agents[2].dead*2 + newstate->agents[3].dead;

                    if(visitedSteps.count(hash) > 0) {
                        delete newstate;
                        continue;
                    }
                    else {
                        visitedSteps.insert(hash);
                    }
#endif

                        Position myNewPos;
                        myNewPos.x = newstate->agents[newstate->ourId].x;
                        myNewPos.y = newstate->agents[newstate->ourId].y;
                        positions_in_chain[depth] = myNewPos;
                        positions_in_chain.count++;

                        StepResult futureSteps;
                        if (depth + 1 < myMaxDepth)
                            futureSteps = runOneStep(newstate, depth + 1);
                        else
                            futureSteps = runAlreadyPlantedBombs(newstate);

                        if ((float)futureSteps > -100 && (float)futureSteps < minPointE2) {
                            minPointE2 = (float)futureSteps;
#ifdef GM_DEBUGMODE_STEPS
                            futureSteps.steps.AddElem(moveE2);
#endif
                            futureStepsE2 = futureSteps;
                        }

                        positions_in_chain.count--;
                        delete newstate;
                    }
                    if (minPointE2 > -100 && minPointE2 < minPointE1) {
                        minPointE1 = minPointE2;
#ifdef GM_DEBUGMODE_STEPS
                        futureStepsE2.steps.AddElem(moveE1);
#endif
                        futureStepsE1 = futureStepsE2;
                    }
                }
                if (minPointE1 < 100 && minPointE1 > maxTeammate) {
                    maxTeammate = minPointE1;
#ifdef GM_DEBUGMODE_STEPS
                    futureStepsE1.steps.AddElem(moveT);
#endif
                    futureStepsT = futureStepsE1;
                }
            }

            moves_in_chain.RemoveAt(moves_in_chain.count - 1);

            if (maxTeammate > -100) {
#ifdef RANDOM_TIEBREAK
                if (maxTeammate == maxPoint && move != 5) { bestmoves[bestmoves.count] = move; bestmoves.count++;}
            if (maxTeammate > maxPoint) { maxPoint = maxTeammate; bestmoves.count = 1; bestmoves[0] = move;}
#else
                if (maxTeammate > (float)stepRes) {
#ifdef GM_DEBUGMODE_STEPS
                    futureStepsT.steps.AddElem(move);
#else
                    if(depth == 0)
                        depth_0_Move = move;
#endif
                    stepRes = futureStepsT;
                }
#endif
            }
        }

#ifdef RANDOM_TIEBREAK
        if(bestmoves.count > 0) {
        if(maxPoint > best_points_in_chain[depth])
        {
            bool foundIdle = false;
            for(int i=0; i<bestmoves.count; i++)
                if(bestmoves[i] == 0) {
                    best_moves_in_chain[depth] = 0;
                    foundIdle = true;
                }
            if(!foundIdle)
                best_moves_in_chain[depth] = bestmoves[(state->timeStep / 4) % bestmoves.count];
            best_points_in_chain[depth] = maxPoint;
        }
    }else
        best_moves_in_chain[depth] = 0;
#endif

        return stepRes;
    }

    void CologneAgent::createDeadEndMap(const State *state) {
        short walkable_neighbours[BOARD_SIZE * BOARD_SIZE];
        memset(walkable_neighbours, 0, BOARD_SIZE * BOARD_SIZE * sizeof(short));
        std::list<Position> deadEnds;
        for (int x = 0; x < BOARD_SIZE; x++) {
            for (int y = 0; y < BOARD_SIZE; y++) {
                if (_CheckPos2(state, x, y)) {
                    walkable_neighbours[x + BOARD_SIZE * y] =
                            (int) _CheckPos2(state, x - 1, y) + (int) _CheckPos2(state, x, y - 1) +
                            (int) _CheckPos2(state, x + 1, y) + (int) _CheckPos2(state, x, y + 1);
                    if (walkable_neighbours[x + BOARD_SIZE * y] < 2) {
                        Position p;
                        p.x = x;
                        p.y = y;
                        deadEnds.push_back(p);
                    }
                }
            }
        }


        memset(leadsToDeadEnd, 0, BOARD_SIZE * BOARD_SIZE * sizeof(bool));

        std::function<void(int, int)> recurseFillWaysToDeadEnds = [&](int x, int y) {
            leadsToDeadEnd[x + BOARD_SIZE * y] = true;

            if (x > 0 && walkable_neighbours[x - 1 + BOARD_SIZE * y] < 2 &&
                leadsToDeadEnd[x - 1 + BOARD_SIZE * y] == false)
                recurseFillWaysToDeadEnds(x - 1, y);
            if (x < BOARD_SIZE - 1 && walkable_neighbours[x + 1 + BOARD_SIZE * y] < 2 &&
                leadsToDeadEnd[x + 1 + BOARD_SIZE * y] == false)
                recurseFillWaysToDeadEnds(x + 1, y);
            if (y > 0 && walkable_neighbours[x + BOARD_SIZE * (y - 1)] < 2 &&
                leadsToDeadEnd[x + BOARD_SIZE * (y - 1)] == false)
                recurseFillWaysToDeadEnds(x, y - 1);
            if (y < BOARD_SIZE - 1 && walkable_neighbours[x + BOARD_SIZE * (y + 1)] < 2 &&
                leadsToDeadEnd[x + BOARD_SIZE * (y + 1)] == false)
                recurseFillWaysToDeadEnds(x, y + 1);
        };

        for (Position p : deadEnds) {
            recurseFillWaysToDeadEnds(p.x, p.y);
        }
    }

    Move CologneAgent::act(const State *state) {
        createDeadEndMap(state);
        visitedSteps.clear();
        simulatedSteps = 0;
        enemyIteration1 = 0;
        enemyIteration2 = 0;
        teammateIteration = 0;
        seenAgents = 0;
        ourId = state->ourId;
        enemy1Id = state->enemy1Id;
        enemy2Id = state->enemy2Id;
        teammateId = state->teammateId;
        positions_in_chain.count = 0;
        int seenEnemies = 0;
        if (!state->agents[teammateId].dead && state->agents[teammateId].x >= 0) {
            seenAgents++;
        }
        if (!state->agents[enemy1Id].dead && state->agents[enemy1Id].x >= 0) {
            seenAgents++;
            seenEnemies++;
        }
        if (!state->agents[enemy2Id].dead && state->agents[enemy2Id].x >= 0) {
            seenAgents++;
            seenEnemies++;
        }

        if (!state->agents[teammateId].dead && state->agents[teammateId].x >= 0) {
            int dist = std::abs(state->agents[ourId].x - state->agents[teammateId].x) +
                       std::abs(state->agents[ourId].y - state->agents[teammateId].y);
            if (dist < 3) teammateIteration++;
            if (dist < 5) teammateIteration++;
        }
        if (!state->agents[enemy1Id].dead && state->agents[enemy1Id].x >= 0) {
            int dist = std::abs(state->agents[ourId].x - state->agents[enemy1Id].x) +
                       std::abs(state->agents[ourId].y - state->agents[enemy1Id].y);
            if (dist < 2) enemyIteration1++;
            if (dist < 3 && seenAgents == 1) enemyIteration1++;
            if (dist < 5) enemyIteration1++;
        }
        if (!state->agents[enemy2Id].dead && state->agents[enemy2Id].x >= 0) {
            int dist = std::abs(state->agents[ourId].x - state->agents[enemy2Id].x) +
                       std::abs(state->agents[ourId].y - state->agents[enemy2Id].y);
            if (dist < 2) enemyIteration2++;
            if (dist < 3 && seenAgents == 1) enemyIteration2++;
            if (dist < 5) enemyIteration2++;
        }
        myMaxDepth = 6 - seenAgents;

        sameAs6_12_turns_ago = true;
        for (int agentId = 0; agentId < 4; agentId++) {
            if (previousPositions[agentId].count == 12) {
                if (previousPositions[agentId][0].x != state->agents[agentId].x ||
                    previousPositions[agentId][0].y != state->agents[agentId].y)
                    sameAs6_12_turns_ago = false;
                if (previousPositions[agentId][6].x != state->agents[agentId].x ||
                    previousPositions[agentId][6].y != state->agents[agentId].y)
                    sameAs6_12_turns_ago = false;
                previousPositions[agentId].RemoveAt(0);
            } else {
                sameAs6_12_turns_ago = false;
            }

            Position p;
            p.x = state->agents[agentId].x;
            p.y = state->agents[agentId].y;
            previousPositions[agentId][previousPositions[agentId].count] = p;
            previousPositions[agentId].count++;
        }
        if (sameAs6_12_turns_ago)
            std::cout << "SAME AS BEFORE!!!!" << std::endl;

        const AgentInfo &a = state->agents[ourId];
        if (state->timeStep > 1 && (expectedPosInNewTurn.x != a.x || expectedPosInNewTurn.y != a.y)) {
            std::cout << "Couldn't move to " << expectedPosInNewTurn.y << ":" << expectedPosInNewTurn.x;
            if (std::abs(state->agents[teammateId].x - expectedPosInNewTurn.x) +
                std::abs(state->agents[teammateId].y - expectedPosInNewTurn.y) == 1)
                std::cout << " - Racing with teammate, probably";
            if (std::abs(state->agents[enemy1Id].x - expectedPosInNewTurn.x) +
                std::abs(state->agents[enemy1Id].y - expectedPosInNewTurn.y) == 1)
                std::cout << " - Racing with enemy1, probably";
            if (std::abs(state->agents[enemy2Id].x - expectedPosInNewTurn.x) +
                std::abs(state->agents[enemy2Id].y - expectedPosInNewTurn.y) == 1)
                std::cout << " - Racing with enemy2, probably";
            std::cout << std::endl;
            lastMoveWasBlocked = true;
            lastBlockedMove = moveHistory[moveHistory.count - 1];
        } else {
            lastMoveWasBlocked = false;
        }

        StepResult stepRes = runOneStep(state, 0);

#ifdef GM_DEBUGMODE_STEPS
        int myMove = stepRes.steps[stepRes.steps.count - 1];
#else
        int myMove = depth_0_Move;
#endif
        if (moveHistory.count == 12) {
            moveHistory.RemoveAt(0);
        }
        moveHistory[moveHistory.count] = myMove;
        moveHistory.count++;

        std::cout << "turn#" << state->timeStep << " ourId:" << ourId << " point: " << (float)stepRes << " selected: ";
        std::cout << myMove << " simulated steps: " << simulatedSteps;
        std::cout << ", depth " << myMaxDepth << " " << teammateIteration << " " << enemyIteration1 << " "
                  << enemyIteration2 << std::endl;
#ifdef GM_DEBUGMODE_STEPS
        for (int i = 0; i < stepRes.steps.count; i++) {
            std::cout << (int) stepRes.steps[stepRes.steps.count - 1 - i] << " ";
            if (i % 4 == 3)
                std::cout << " | ";
        }
#endif

#ifdef GM_DEBUGMODE_COMMENTS
        std::cout << stepRes.comment << std::endl;
#else
        std::cout << std::endl;
#endif

        totalSimulatedSteps += simulatedSteps;
        turns++;
        expectedPosInNewTurn = bboard::util::DesiredPosition(a.x, a.y, (bboard::Move) myMove);
        return (bboard::Move) myMove;
    }

    void CologneAgent::PrintDetailedInfo() {
    }

}
