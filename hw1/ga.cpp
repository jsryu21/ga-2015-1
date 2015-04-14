#include <cstdlib>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include <iterator>
#include <set>
#include "test_case.h"

// https://isocpp.org/wiki/faq/pointers-to-members
#define CALL_MEMBER_FN(object,ptrToMember)  ((object).*(ptrToMember))

const static int PSIZE = 100; // Size of the population
const static int ROULETTE_SELECTION_PRESSURE_K = 3; // 3 ~ 4
const static double TOURNAMENT_SELECTION_PRESSURE_T = 0.5;
// http://www.complex-systems.com/pdf/09-3-2.pdf
// http://en.wikipedia.org/wiki/Tournament_selection
const static int GENERAL_TOURNAMENT_SELECTION_PRESSURE_K = 5;
const static double RANK_SELECTION_PRESSURE_MAX = 3;
const static double RANK_SELECTION_PRESSURE_MIN = 1;
const static double HYBRID_REPLACEMENT_T = 0.8;
const static double TYPICAL_MUTATION_THRESHOLD = 0.5;

struct Solution
{
    Solution(int len);
    std::vector< int > Chromosome;
    double Fitness;
    bool operator <(const Solution& solution) const {
        return Fitness < solution.Fitness;
    }
    bool operator== (const Solution& solution) const {
        return std::equal(Chromosome.begin(), Chromosome.end(), solution.Chromosome.begin());
    }
    friend std::ostream& operator<<(std::ostream& os, const Solution& solution);
};

Solution::Solution(int len) : Chromosome(std::vector< int >(len)),
    Fitness(std::numeric_limits< double >::max())
{
}

std::ostream& operator<<(std::ostream& os, const Solution& solution) {
    for (int i = 0; i < solution.Chromosome.size(); ++i) {
        os << solution.Chromosome[i] << " ";
    }
    os << ": " << solution.Fitness;
    return os;
}

class SteadyStateGA {
    /*****************************************************************
      GA variables and functions
      Note that the representation is currently order-based.
      A chromosome will be a permutation of (0, 1, ..., N-1).
     *****************************************************************/
    public:
        typedef void (SteadyStateGA::*EvaluateFn)(Solution& s);
        typedef void (SteadyStateGA::*GenerateRandomSolutionFn)(Solution& s);
        typedef void (SteadyStateGA::*PreprocessFn)();
        typedef void (SteadyStateGA::*SelectionFn)(Solution& s);
        typedef void (SteadyStateGA::*CrossoverFn)(const Solution& p1, const Solution& p2, Solution& c);
        typedef void (SteadyStateGA::*MutationFn)(Solution& s);
        typedef void (SteadyStateGA::*ReplacementFn)(const Solution& p1, const Solution& p2, const Solution& s);
        SteadyStateGA(const TestCase& testCase
                , EvaluateFn Evaluate_
                , GenerateRandomSolutionFn GenerateRandomSolution_
                , PreprocessFn Preprocess_
                , SelectionFn Selection_
                , CrossoverFn Crossover_
                , MutationFn Mutation_
                , ReplacementFn Replacement_);
        void Evaluate1(Solution& s);
        void GenerateRandomSolution1(Solution& s);
        void Preprocess1();
        void Preprocess2();
        void Selection1(Solution& s);
        void Selection2(Solution& s);
        void Selection3(Solution& s);
        void Selection4(Solution& s);
        void Crossover1(const Solution& p1, const Solution& p2, Solution& c);
        void Crossover2(const Solution& p1, const Solution& p2, Solution& c);
        void Crossover3(const Solution& p1, const Solution& p2, Solution& c);
        void Crossover4(const Solution& p1, const Solution& p2, Solution& c);
        void Crossover5(const Solution& p1, const Solution& p2, Solution& c);
        void Crossover6(const Solution& p1, const Solution& p2, Solution& c);
        void Mutation1(Solution& s);
        void Mutation2(Solution& s);
        void Mutation3(Solution& s);
        void Mutation4(Solution& s);
        void Mutation5(Solution& s);
        void Mutation6(Solution& s);
        void Mutation7(Solution& s);
        void Mutation8(Solution& s);
        void Replacement1(const Solution& p1, const Solution& p2, const Solution& offspr);
        void Replacement2(const Solution& p1, const Solution& p2, const Solution& offspr);
        void Replacement3(const Solution& p1, const Solution& p2, const Solution& offspr);
        void Replacement4(const Solution& p1, const Solution& p2, const Solution& offspr);
        void Replacement5(const Solution& p1, const Solution& p2, const Solution& offspr);
        void Replacement6(const Solution& p1, const Solution& p2, const Solution& offspr);
        void GA();
        void Answer();
        void PrintAllSolutions();
    private:
        EvaluateFn Evaluate;
        GenerateRandomSolutionFn GenerateRandomSolution;
        PreprocessFn Preprocess;
        SelectionFn Selection;
        CrossoverFn Crossover;
        MutationFn Mutation;
        ReplacementFn Replacement;
        void Normalize(Solution& s);

        int solutionLen;
        const std::vector< std::vector< double > >& solutionDist;
        long long timeLimit;
        // population of solutions
        std::vector< Solution > population;
        Solution record;
        Solution randomSolution;
        Solution tempSolution;
        std::vector< double > adjustedFitnesses;
        double sumOfFitnesses;
};

SteadyStateGA::SteadyStateGA(const TestCase& testCase
        , EvaluateFn Evaluate_
        , GenerateRandomSolutionFn GenerateRandomSolution_
        , PreprocessFn Preprocess_
        , SelectionFn Selection_
        , CrossoverFn Crossover_
        , MutationFn Mutation_
        , ReplacementFn Replacement_) : solutionLen(testCase.NumLocations),
    solutionDist(testCase.Dist),
    timeLimit(testCase.TimeLimit),
    population(PSIZE, Solution(solutionLen)),
    record(solutionLen),
    randomSolution(solutionLen),
    tempSolution(solutionLen),
    adjustedFitnesses(PSIZE),
    sumOfFitnesses(0),
    Evaluate(Evaluate_),
    GenerateRandomSolution(GenerateRandomSolution_),
    Preprocess(Preprocess_),
    Selection(Selection_),
    Crossover(Crossover_),
    Mutation(Mutation_),
    Replacement(Replacement_)
{
    for (int i = 0; i < solutionLen; ++i) {
        randomSolution.Chromosome[i] = i;
    }
}

// calculate the fitness of s and store it into s->f
void SteadyStateGA::Evaluate1(Solution& s) {
    s.Fitness = 0;
    for (int i = 0; i < solutionLen; ++i) {
        s.Fitness += solutionDist[s.Chromosome[i]][s.Chromosome[(i + 1) % solutionLen]];
    }
    if (s.Fitness < record.Fitness) {
        record = s;
    }
}

// generate a random order-based solution at s
void SteadyStateGA::GenerateRandomSolution1(Solution& s) {
    for (int i = 0; i < solutionLen; ++i) {
        int r = std::rand() % solutionLen;
        std::swap(randomSolution.Chromosome[i], randomSolution.Chromosome[r]);
    }
    std::vector< int >::iterator zeroIter = std::find(randomSolution.Chromosome.begin(), randomSolution.Chromosome.end(), 0);
    std::copy(randomSolution.Chromosome.begin(), zeroIter, std::copy(zeroIter, randomSolution.Chromosome.end(), s.Chromosome.begin()));
    // calculate the fitness
    CALL_MEMBER_FN(*this, Evaluate)(s);
}

void SteadyStateGA::Preprocess1() {
    double maxFitness = 0;
    double minFitness = std::numeric_limits< double >::max();
    for (int i = 0; i < PSIZE; ++i) {
        maxFitness = std::max(maxFitness, population[i].Fitness);
        minFitness = std::min(minFitness, population[i].Fitness);
    }
    sumOfFitnesses = 0;
    for (int i = 0; i < PSIZE; ++i) {
        double adjustedFitness = (maxFitness - population[i].Fitness) + (maxFitness - minFitness) / (ROULETTE_SELECTION_PRESSURE_K - 1);
        sumOfFitnesses += adjustedFitness;
        adjustedFitnesses[i] = adjustedFitness;
    }
}

void SteadyStateGA::Preprocess2() {
    std::sort(population.begin(), population.end());
    sumOfFitnesses = 0;
    for (int i = 0; i < PSIZE; ++i) {
        double adjustedFitness = RANK_SELECTION_PRESSURE_MAX + i * (RANK_SELECTION_PRESSURE_MIN - RANK_SELECTION_PRESSURE_MAX) / (PSIZE - 1);
        sumOfFitnesses += adjustedFitness;
        adjustedFitnesses[i] = adjustedFitness;
    }
}

// choose one solution from the population
// currently this operator randomly chooses one w/ uniform Distribution
void SteadyStateGA::Selection1(Solution& p) {
    int r = std::rand() % PSIZE;
    p = population[r];
}

// Roulette Wheel - Preprocess1
// Rank - Preprocess2
void SteadyStateGA::Selection2(Solution& p) {
    double point = static_cast< double >(std::rand()) * sumOfFitnesses / RAND_MAX;
    double sum = 0;
    for (int i = 0; i < PSIZE; ++i) {
        sum += adjustedFitnesses[i];
        if (point < sum) {
            p = population[i];
            return;
        }
    }
    // http://valgrind.org/gallery/linux_mag.html
    p = population[PSIZE - 1];
}

// Tournament
void SteadyStateGA::Selection3(Solution& p) {
    int r1 = std::rand() % PSIZE;
    int r2 = std::rand() % PSIZE;
    const Solution& p1 = population[r1];
    const Solution& p2 = population[r2];
    double r = static_cast< double >(std::rand()) / RAND_MAX;
    if (TOURNAMENT_SELECTION_PRESSURE_T > r) {
        p = (p1.Fitness >= p2.Fitness) ? p1 : p2;
    } else {
        p = (p1.Fitness >= p2.Fitness) ? p2 : p1;
    }
}

// General Tournament
void SteadyStateGA::Selection4(Solution& p) {
    std::vector< Solution > tournament;
    for (int i = 0; i < GENERAL_TOURNAMENT_SELECTION_PRESSURE_K; ++i) {
        int r = std::rand() % PSIZE;
        tournament.push_back(population[r]);
    }
    std::sort(tournament.begin(), tournament.end());
    double r = static_cast< double >(std::rand()) / RAND_MAX;
    for (int i = 0; i < GENERAL_TOURNAMENT_SELECTION_PRESSURE_K; ++i) {
        if (TOURNAMENT_SELECTION_PRESSURE_T * std::pow((1 - TOURNAMENT_SELECTION_PRESSURE_T), i) < r) {
            p = tournament[i];
            return;
        }
    }
    p = tournament[GENERAL_TOURNAMENT_SELECTION_PRESSURE_K - 1];
}

// combine the given parents p1 and p2
// and store the generated solution at c
// currently the child will be same as p1
void SteadyStateGA::Crossover1(const Solution& p1, const Solution& p2, Solution& c) {
    int point = std::rand() % solutionLen;
    std::vector< int >::const_iterator pointIter = p1.Chromosome.begin() + point;
    std::copy(p1.Chromosome.begin(), pointIter, c.Chromosome.begin());
    int index = point;
    for (int i = point; i < solutionLen; ++i) {
        if (std::find(p1.Chromosome.begin(), pointIter, p2.Chromosome[i]) == pointIter) {
            c.Chromosome[index] = p2.Chromosome[i];
            index++;
        }
    }
    for (int i = 0; i < point; ++i) {
        if (std::find(p1.Chromosome.begin(), pointIter, p2.Chromosome[i]) == pointIter) {
            c.Chromosome[index] = p2.Chromosome[i];
            index++;
        }
    }
    Normalize(c);
    CALL_MEMBER_FN(*this, Evaluate)(c);
}

// order crossover
void SteadyStateGA::Crossover2(const Solution& p1, const Solution& p2, Solution& c) {
    int p = std::rand() % solutionLen;
    int q = std::rand() % solutionLen;
    while (p == q) {
        q = std::rand() % solutionLen;
    }
    if (p > q) {
        std::swap(p, q);
    }
    std::vector< int >::const_iterator pIter = p1.Chromosome.begin() + p;
    std::vector< int >::const_iterator qIter = p1.Chromosome.begin() + q;
    std::copy(pIter, qIter, c.Chromosome.begin() + p);
    int index = 0;
    for (int i = q; i < solutionLen; ++i) {
        if (std::find(pIter, qIter, p2.Chromosome[i]) == qIter) {
            if (index < p) {
                c.Chromosome[index] = p2.Chromosome[i];
                index++;
            } else if (index < q) {
                c.Chromosome[q] = p2.Chromosome[i];
                index = q + 1;
            } else {
                c.Chromosome[index] = p2.Chromosome[i];
                index++;
            }
        }
    }
    for (int i = 0; i < q; ++i) {
        if (std::find(pIter, qIter, p2.Chromosome[i]) == qIter) {
            if (index < p) {
                c.Chromosome[index] = p2.Chromosome[i];
                index++;
            } else if (index < q) {
                c.Chromosome[q] = p2.Chromosome[i];
                index = q + 1;
            } else {
                c.Chromosome[index] = p2.Chromosome[i];
                index++;
            }
        }
    }
    Normalize(c);
    CALL_MEMBER_FN(*this, Evaluate)(c);
}

// cycle crossover
void SteadyStateGA::Crossover3(const Solution& p1, const Solution& p2, Solution& c) {
    int turn = 0;
    std::vector< bool > flags(solutionLen, false);
    std::vector< bool >::iterator iter = std::find(flags.begin(), flags.end(), false);
    while (iter != flags.end()) {
        if (turn == 0) {
            int index = std::distance(flags.begin(), iter);
            int p1Gene = p1.Chromosome[index];
            int p2Gene = p2.Chromosome[index];
            c.Chromosome[index] = p1.Chromosome[index];
            flags[index] = true;
            while (p1Gene != p2Gene) {
                index = std::distance(p1.Chromosome.begin(), std::find(p1.Chromosome.begin(), p1.Chromosome.end(), p2Gene));
                c.Chromosome[index] = p1.Chromosome[index];
                p2Gene = p2.Chromosome[index];
                flags[index] = true;
            }
            iter = std::find(flags.begin(), flags.end(), false);
        } else {
            int index = std::distance(flags.begin(), iter);
            int p2Gene = p2.Chromosome[index];
            int p1Gene = p1.Chromosome[index];
            c.Chromosome[index] = p2.Chromosome[index];
            flags[index] = true;
            while (p2Gene != p1Gene) {
                index = std::distance(p2.Chromosome.begin(), std::find(p2.Chromosome.begin(), p2.Chromosome.end(), p1Gene));
                c.Chromosome[index] = p2.Chromosome[index];
                p1Gene = p1.Chromosome[index];
                flags[index] = true;
            }
            iter = std::find(flags.begin(), flags.end(), false);
        }
    }
    Normalize(c);
    CALL_MEMBER_FN(*this, Evaluate)(c);
}

// PMX : partially matched crossover
void SteadyStateGA::Crossover4(const Solution& p1, const Solution& p2, Solution& c) {
    int p = std::rand() % solutionLen;
    int q = std::rand() % solutionLen;
    while (p == q) {
        q = std::rand() % solutionLen;
    }
    if (p > q) {
        std::swap(p, q);
    }
    std::vector< int >::const_iterator p1PIter = p1.Chromosome.begin() + p;
    std::vector< int >::const_iterator p1QIter = p1.Chromosome.begin() + q;
    std::vector< int >::const_iterator p2PIter = p2.Chromosome.begin() + p;
    std::vector< int >::const_iterator p2QIter = p2.Chromosome.begin() + q;
    std::vector< int >::iterator cIter = std::copy(p2.Chromosome.begin(), p2PIter, c.Chromosome.begin());
    cIter = std::copy(p1PIter, p1QIter, cIter);
    std::copy(p2QIter, p2.Chromosome.end(), cIter);
    for (int i = 0; i < p; ++i) {
        int cGene = c.Chromosome[i];
        int p2Gene = p2.Chromosome[i];
        std::vector< int >::const_iterator p1Iter = std::find(p1PIter, p1QIter, cGene);
        while (p1Iter != p1QIter) {
            p2Gene = p2.Chromosome[std::distance(p1.Chromosome.begin(), p1Iter)];
            p1Iter = std::find(p1PIter, p1QIter, p2Gene);
        }
        c.Chromosome[i] = p2Gene;
    }
    for (int i = q; i < solutionLen; ++i) {
        int cGene = c.Chromosome[i];
        int p2Gene = p2.Chromosome[i];
        std::vector< int >::const_iterator p1Iter = std::find(p1PIter, p1QIter, cGene);
        while (p1Iter != p1QIter) {
            p2Gene = p2.Chromosome[std::distance(p1.Chromosome.begin(), p1Iter)];
            p1Iter = std::find(p1PIter, p1QIter, p2Gene);
        }
        c.Chromosome[i] = p2Gene;
    }
    Normalize(c);
    CALL_MEMBER_FN(*this, Evaluate)(c);
}

// edge recombination
// http://www.rubicite.com/Tutorials/GeneticAlgorithms/CrossoverOperators/EdgeRecombinationCrossoverOperator.aspx
void SteadyStateGA::Crossover5(const Solution& p1, const Solution& p2, Solution& c) {
    std::vector< std::set< int > > neighborList(solutionLen);
    for (int i = 0; i < solutionLen; ++i) {
        neighborList[i].insert(p1.Chromosome[(i + solutionLen - 1) % solutionLen]);
        neighborList[i].insert(p1.Chromosome[(i + 1) % solutionLen]);
        neighborList[i].insert(p2.Chromosome[(i + solutionLen - 1) % solutionLen]);
        neighborList[i].insert(p2.Chromosome[(i + 1) % solutionLen]);
    }
    int city = 0;
    int index = 0;
    while (true) {
        c.Chromosome[index++] = city;
        if (index >= solutionLen) {
            break;
        }
        for (int i = 0; i < solutionLen; ++i) {
            neighborList[i].erase(city);
        }
        if (neighborList[city].size() == 0) {
            // random node not already in CHILD
            city = std::rand() % solutionLen;
            while (std::find(c.Chromosome.begin(), c.Chromosome.begin() + index, city) != c.Chromosome.begin() + index) {
                city = std::rand() % solutionLen;
            }
        } else {
            int fewest = 5;
            std::set< int > neighbors = neighborList[city];
            for (std::set< int >::iterator iter = neighbors.begin(); iter != neighbors.end(); ++iter) {
                if (neighborList[*iter].size() < fewest) {
                    city = *iter;
                    fewest = neighborList[*iter].size();
                }
            }
        }
    }
    Normalize(c);
    CALL_MEMBER_FN(*this, Evaluate)(c);
}

// hybrid
void SteadyStateGA::Crossover6(const Solution& p1, const Solution& p2, Solution& c) {
    Normalize(c);
    CALL_MEMBER_FN(*this, Evaluate)(c);
}

// mutate the solution s
// two-swap or swap-change
void SteadyStateGA::Mutation1(Solution& s) {
    int p = std::rand() % solutionLen;
    int q = std::rand() % solutionLen;
    std::swap(s.Chromosome[p], s.Chromosome[q]); // swap
    Normalize(s);
    CALL_MEMBER_FN(*this, Evaluate)(s);
}

// typical mutation for tsp
void SteadyStateGA::Mutation2(Solution& s) {
    std::vector< int > mutatedGenes;
    std::vector< int > mutatedGenesIndex;
    for (int i = 0; i < solutionLen; ++i) {
        double r = static_cast< double >(std::rand()) / RAND_MAX;
        if (r < TYPICAL_MUTATION_THRESHOLD) {
            mutatedGenes.push_back(s.Chromosome[i]);
            mutatedGenesIndex.push_back(i);
        }
    }
    std::random_shuffle(mutatedGenes.begin(), mutatedGenes.end());
    for (int i = 0; i < mutatedGenesIndex.size(); ++i) {
        s.Chromosome[mutatedGenesIndex[i]] = mutatedGenes[i];
    }
    Normalize(s);
    CALL_MEMBER_FN(*this, Evaluate)(s);
}

// range shuffle
void SteadyStateGA::Mutation3(Solution& s) {
    int p = std::rand() % solutionLen;
    int q = std::rand() % solutionLen;
    if (p > q) {
        std::swap(p, q);
    }
    std::random_shuffle(s.Chromosome.begin() + p, s.Chromosome.begin() + q);
    Normalize(s);
    CALL_MEMBER_FN(*this, Evaluate)(s);
}

// inversion == 2-change(two-change)
void SteadyStateGA::Mutation4(Solution& s) {
    int p = std::rand() % solutionLen;
    int q = std::rand() % solutionLen;
    if (p > q) {
        std::swap(p, q);
    }
    std::reverse(s.Chromosome.begin() + p, s.Chromosome.begin() + q);
    Normalize(s);
    CALL_MEMBER_FN(*this, Evaluate)(s);
}

// double bridge kick move
void SteadyStateGA::Mutation5(Solution& s) {
    if (solutionLen < 4) {
        return;
    }
    std::vector< int > rs;
    for (int i = 0; i < 3; ++i) {
        int r = std::rand() % (solutionLen - 1);
        while (std::find(rs.begin(), rs.end(), r) != rs.end()) {
            r = std::rand() % (solutionLen - 1);
        }
        rs.push_back(r + 1);
    }
    std::sort(rs.begin(), rs.end());
    std::vector< int > mutatedGenes(solutionLen);
    std::vector< int >::iterator iter = std::copy(s.Chromosome.begin() + rs[2], s.Chromosome.end(), mutatedGenes.begin());
    iter = std::copy(s.Chromosome.begin() + rs[1], s.Chromosome.begin() + rs[2], iter);
    iter = std::copy(s.Chromosome.begin() + rs[0], s.Chromosome.begin() + rs[1], iter);
    std::copy(s.Chromosome.begin(), s.Chromosome.begin() + rs[0], iter);
    std::copy(mutatedGenes.begin(), mutatedGenes.end(), s.Chromosome.begin());
    Normalize(s);
    CALL_MEMBER_FN(*this, Evaluate)(s);
}

// hybrid - double bridge kick move : inversion = 1 : 9
void SteadyStateGA::Mutation6(Solution& s) {
    int r = std::rand() % 10;
    if (r < 1) {
        Mutation5(s);
    } else {
        Mutation4(s);
    }
}

// or change
void SteadyStateGA::Mutation7(Solution& s) {
    int p = std::rand() % solutionLen;
    int q = std::rand() % solutionLen;
    while (p == q) {
        q = std::rand() % solutionLen;
    }
    if (p > q) {
        std::swap(p, q);
    }
    int gene = s.Chromosome[p];
    std::rotate(s.Chromosome.begin() + p, s.Chromosome.begin() + p + 1, s.Chromosome.begin() + q + 1);
    s.Chromosome[q] = gene;
    Normalize(s);
    CALL_MEMBER_FN(*this, Evaluate)(s);
}

// fully hybrid
void SteadyStateGA::Mutation8(Solution& s) {
    int r = std::rand() % 6;
    if (r == 0) {
        Mutation1(s);
    } else if (r == 1) {
        Mutation2(s);
    } else if (r == 2) {
        Mutation3(s);
    } else if (r == 3) {
        Mutation4(s);
    } else if (r == 4) {
        Mutation5(s);
    } else if (r == 5) {
        Mutation7(s);
    }
}

// replace one solution from the population with the new offspring
// currently any random solution can be replaced
void SteadyStateGA::Replacement1(const Solution& p1, const Solution& p2, const Solution& offspr) {
    int p = std::rand() % PSIZE;
    population[p] = offspr;
}

// elitism
void SteadyStateGA::Replacement2(const Solution& p1, const Solution& p2, const Solution& offspr) {
    std::sort(population.begin(), population.end());
    population.back() = offspr;
}

// preselection
void SteadyStateGA::Replacement3(const Solution& p1, const Solution& p2, const Solution& offspr) {
    std::vector< Solution >::iterator iter = std::find(population.begin(), population.end(), (p1.Fitness > p2.Fitness) ? p1 : p2);
    if (iter != population.end()) {
        *iter = offspr;
    }
}

void SteadyStateGA::Replacement4(const Solution& p1, const Solution& p2, const Solution& offspr) {
    if (offspr.Fitness < p1.Fitness || offspr.Fitness < p2.Fitness) {
        Replacement3(p1, p2, offspr);
    } else {
        Replacement2(p1, p2, offspr);
    }
}

void SteadyStateGA::Replacement5(const Solution& p1, const Solution& p2, const Solution& offspr) {
    if (offspr.Fitness < p1.Fitness || offspr.Fitness < p2.Fitness) {
        Replacement3(p1, p2, offspr);
    }
}

void SteadyStateGA::Replacement6(const Solution& p1, const Solution& p2, const Solution& offspr) {
    double r = static_cast< double >(std::rand()) / RAND_MAX;
    if (r < HYBRID_REPLACEMENT_T) {
        Replacement3(p1, p2, offspr);
    } else {
        Replacement2(p1, p2, offspr);
    }
}

// a "steady-state" GA
void SteadyStateGA::GA() {
    std::time_t begin = std::time(0);
    for (int i = 0; i < PSIZE; ++i) {
        CALL_MEMBER_FN(*this, GenerateRandomSolution)(population[i]);
    }
    while (true) {
        if (std::time(0) - begin > timeLimit - 1) {
            return; // end condition
        }
        Solution p1(solutionLen);
        Solution p2(solutionLen);
        Solution c(solutionLen);
        if (Preprocess) {
            CALL_MEMBER_FN(*this, Preprocess)();
        }
        CALL_MEMBER_FN(*this, Selection)(p1);
        CALL_MEMBER_FN(*this, Selection)(p2);
        CALL_MEMBER_FN(*this, Crossover)(p1, p2, c);
        CALL_MEMBER_FN(*this, Mutation)(c);
        CALL_MEMBER_FN(*this, Replacement)(p1, p2, c);
    }
}

// print the best solution found to stdout
void SteadyStateGA::Answer() {
    std::cout << record << std::endl;
}

void SteadyStateGA::PrintAllSolutions() {
    for (int i = 0; i < PSIZE; ++i) {
        std::cout << population[i] << std::endl;
    }
}

void SteadyStateGA::Normalize(Solution& s) {
    std::copy(s.Chromosome.begin(), s.Chromosome.end(), tempSolution.Chromosome.begin());
    std::vector< int >::iterator zeroIter = std::find(tempSolution.Chromosome.begin(), tempSolution.Chromosome.end(), 0);
    std::copy(tempSolution.Chromosome.begin(), zeroIter, std::copy(zeroIter, tempSolution.Chromosome.end(), s.Chromosome.begin()));
}

int main() {
    // http://en.cppreference.com/w/cpp/numeric/random/rand
    std::srand(std::time(0));
    TestCase testCase;
    //testCase.PrintTestCase();
    SteadyStateGA ga(testCase
            , &SteadyStateGA::Evaluate1
            , &SteadyStateGA::GenerateRandomSolution1
            , &SteadyStateGA::Preprocess1
            , &SteadyStateGA::Selection2
            , &SteadyStateGA::Crossover5
            , &SteadyStateGA::Mutation8
            , &SteadyStateGA::Replacement6);
    ga.GA();
    ga.Answer();
    //ga.PrintAllSolutions();
    return 0;
}
