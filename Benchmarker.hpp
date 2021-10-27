#ifndef BENCHMARKING_HPP
#define BENCHMARKING_HPP

#include <iostream>
#include <chrono>
#include <map>
#include <vector>
#include <string>

#include "CSV_Logger.hpp"

class Benchmarker
{
private:
	std::map<std::string, double> startTimers;
	std::map<std::string, double> endTimers;
	std::map<std::string, double> lastTimers;
	std::map<std::string, double> totalTimers;
	std::map<std::string, double> maxDurations;
	std::map<std::string, double> minDurations;
	std::map<std::string, double> maxDifference;
	std::map<std::string, double> averageDifference;
	std::map<std::string, double> lastElapsedTime;

	std::map<std::string, uint32_t> cntTimersAverage;

	CSV_Logger logger_;
public:
	Benchmarker(const std::string aPath, const std::vector<std::string> aFields) : logger_(aPath, aFields)
	{
	}
	//Default functions which use std::chrono to take times.
	void startTimer(const std::string aTimer)
	{
		if (cntTimersAverage[aTimer] == 0)
		{
			maxDifference[aTimer] = 0.0;
			maxDurations[aTimer] = 0.0;
			minDurations[aTimer] = 9999999.0;
		}

		//Start timer and increment number of timers//
		++cntTimersAverage[aTimer];
		startTimers[aTimer] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
	}
	void waitTimer(const std::string aTimer)
	{
		//Calculate total time//
		endTimers[aTimer] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
		totalTimers[aTimer] += endTimers[aTimer] - startTimers[aTimer];
	}
	void resumeTimer(const std::string aTimer)
	{
		startTimers[aTimer] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
	}
	void pauseTimer(const std::string aTimer)
	{
		//Calculate total time//
		endTimers[aTimer] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
		totalTimers[aTimer] += endTimers[aTimer] - startTimers[aTimer];

		//Calculate differences//
		double elapsedTime = endTimers[aTimer] - startTimers[aTimer];
		double difference = abs(elapsedTime - lastElapsedTime[aTimer]);
		averageDifference[aTimer] += difference;
		lastElapsedTime[aTimer] = elapsedTime;

		//Set max and mins//
		maxDifference[aTimer] = difference > maxDifference[aTimer] ? difference : maxDifference[aTimer];
		maxDurations[aTimer] = elapsedTime > maxDurations[aTimer] ? elapsedTime : maxDurations[aTimer];
		minDurations[aTimer] = elapsedTime < minDurations[aTimer] ? elapsedTime : minDurations[aTimer];
	}
	void endTimer(const std::string aTimer)
	{
		endTimers[aTimer] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
		totalTimers[aTimer] += endTimers[aTimer] - startTimers[aTimer];
	}
	//Overloading functions which accepted collected timestamps.
	void startTimer(const std::string aTimer, double aTimestamp)
	{
		if (cntTimersAverage[aTimer] == 0)
		{
			maxDifference[aTimer] = 0.0;
			maxDurations[aTimer] = 0.0;
			minDurations[aTimer] = 9999999.0;
		}

		//Start timer and increment number of timers//
		++cntTimersAverage[aTimer];
		startTimers[aTimer] = aTimestamp;
	}
	void pauseTimer(const std::string aTimer, double aTimestamp)
	{
		//Calculate total time//
		endTimers[aTimer] = aTimestamp;
		totalTimers[aTimer] += endTimers[aTimer] - startTimers[aTimer];

		//Calculate differences//
		double elapsedTime = endTimers[aTimer] - startTimers[aTimer];
		double difference = abs(elapsedTime - lastElapsedTime[aTimer]);
		averageDifference[aTimer] += difference;
		lastElapsedTime[aTimer] = elapsedTime;

		//Set max and mins//
		maxDifference[aTimer] = difference > maxDifference[aTimer] ? difference : maxDifference[aTimer];
		maxDurations[aTimer] = elapsedTime > maxDurations[aTimer] ? elapsedTime : maxDurations[aTimer];
		minDurations[aTimer] = elapsedTime < minDurations[aTimer] ? elapsedTime : minDurations[aTimer];
	}
	void addTimer(const std::string aTimer, double aTimestamp)
	{
		if (cntTimersAverage[aTimer] == 0)
		{
			maxDifference[aTimer] = 0.0;
			maxDurations[aTimer] = 0.0;
			minDurations[aTimer] = 9999999.0;
		}
		++cntTimersAverage[aTimer];
		totalTimers[aTimer] += aTimestamp;

		//Calculate differences//
		double elapsedTime = aTimestamp;
		double difference = abs(elapsedTime - lastElapsedTime[aTimer]);
		averageDifference[aTimer] += difference;
		lastElapsedTime[aTimer] = elapsedTime;

		//Set max and mins//
		maxDifference[aTimer] = difference > maxDifference[aTimer] ? difference : maxDifference[aTimer];
		maxDurations[aTimer] = elapsedTime > maxDurations[aTimer] ? elapsedTime : maxDurations[aTimer];
		minDurations[aTimer] = elapsedTime < minDurations[aTimer] ? elapsedTime : minDurations[aTimer];
	}

	void elapsedTimer(const std::string aTimer)
	{
		std::vector<std::string> record;
		record.push_back(aTimer);
		record.push_back(std::to_string(totalTimers[aTimer]));

		//auto diff = end - start;
		std::cout << "Benchmarker: " << aTimer << std::endl;
		std::cout << "Total time to complete: " << totalTimers[aTimer] / (float)1e3 << "s" << std::endl;
		std::cout << "Total time to complete: " << totalTimers[aTimer] << "ms" << std::endl;
		std::cout << "Total time to complete: " << totalTimers[aTimer] * (float)1e6 << "ns" << std::endl;

		//Calculate average time per buffer & average difference//
		if (cntTimersAverage[aTimer] > 1)
		{
			double avgElapsed = std::chrono::duration <double, std::milli>(totalTimers[aTimer]).count() / cntTimersAverage[aTimer];
			std::cout << "Average time to complete each buffer: " << avgElapsed << "ms" << std::endl << std::endl;
			record.push_back(std::to_string(avgElapsed));
			averageDifference[aTimer] = averageDifference[aTimer] / cntTimersAverage[aTimer];
		}

		record.push_back(std::to_string(maxDurations[aTimer]));
		record.push_back(std::to_string(minDurations[aTimer]));
		record.push_back(std::to_string(maxDifference[aTimer]));
		record.push_back(std::to_string(averageDifference[aTimer]));
		logger_.addRecord(record);

		//Reset timers//
		cntTimersAverage[aTimer] = 0;
		totalTimers[aTimer] = 0.0;
	}
	bool close()
	{
		logger_.close();
		return true;
	}
};

#endif