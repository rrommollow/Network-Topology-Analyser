#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <queue>
#include <set>
#include <limits>
#include <algorithm>
#include <memory>
#include <functional>
#include <random>
#include <sstream>
#include <iomanip>
#include <fstream>  // Added missing include

// Forward declarations
class Device;
class Connection;
class Network;

// Enums for device types and connection types
enum class DeviceType { ROUTER, SWITCH, SERVER, CLIENT, FIREWALL };
enum class ConnectionType { COPPER, FIBER, WIRELESS };

// Utility function to convert enums to string
std::string deviceTypeToString(DeviceType type) {
    switch (type) {
        case DeviceType::ROUTER: return "Router";
        case DeviceType::SWITCH: return "Switch";
        case DeviceType::SERVER: return "Server";
        case DeviceType::CLIENT: return "Client";
        case DeviceType::FIREWALL: return "Firewall";
        default: return "Unknown";
    }
}

std::string connectionTypeToString(ConnectionType type) {
    switch (type) {
        case ConnectionType::COPPER: return "Copper";
        case ConnectionType::FIBER: return "Fiber";
        case ConnectionType::WIRELESS: return "Wireless";
        default: return "Unknown";
    }
}

//=============================================================================
// 1. Device and Connection Classes
//=============================================================================

class Device {
private:
    int id;
    std::string name;
    DeviceType type;
    double processingCapacity; // Measured in packets per second
    bool isActive;
    std::map<std::string, std::string> properties;

public:
    Device(int id, const std::string& name, DeviceType type, double processingCapacity)
        : id(id), name(name), type(type), processingCapacity(processingCapacity), isActive(true) {}

    int getId() const { return id; }
    std::string getName() const { return name; }
    DeviceType getType() const { return type; }
    double getProcessingCapacity() const { return processingCapacity; }
    bool getIsActive() const { return isActive; }

    void setActive(bool active) { isActive = active; }
    void setProcessingCapacity(double capacity) { processingCapacity = capacity; }
    
    void setProperty(const std::string& key, const std::string& value) {
        properties[key] = value;
    }
    
    std::string getProperty(const std::string& key) const {
        auto it = properties.find(key);
        return (it != properties.end()) ? it->second : "";
    }
    
    std::string toString() const {
        std::stringstream ss;
        ss << "Device #" << id << " (" << name << "): " 
           << deviceTypeToString(type) 
           << ", Capacity: " << processingCapacity
           << ", Status: " << (isActive ? "Active" : "Inactive");
        return ss.str();
    }
};

class Connection {
private:
    int id;
    int sourceId;
    int destId;
    ConnectionType type;
    double bandwidth;    // Measured in Mbps
    double latency;      // Measured in ms
    double reliability;  // From 0 to 1
    bool isActive;
    std::map<std::string, std::string> properties;

public:
    Connection(int id, int sourceId, int destId, ConnectionType type, 
               double bandwidth, double latency, double reliability)
        : id(id), sourceId(sourceId), destId(destId), type(type),
          bandwidth(bandwidth), latency(latency), reliability(reliability), isActive(true) {}

    int getId() const { return id; }
    int getSourceId() const { return sourceId; }
    int getDestId() const { return destId; }
    ConnectionType getType() const { return type; }
    double getBandwidth() const { return bandwidth; }
    double getLatency() const { return latency; }
    double getReliability() const { return reliability; }
    bool getIsActive() const { return isActive; }

    void setActive(bool active) { isActive = active; }
    void setBandwidth(double bw) { bandwidth = bw; }
    void setLatency(double lat) { latency = lat; }
    void setReliability(double rel) { reliability = rel; }
    
    void setProperty(const std::string& key, const std::string& value) {
        properties[key] = value;
    }
    
    std::string getProperty(const std::string& key) const {
        auto it = properties.find(key);
        return (it != properties.end()) ? it->second : "";
    }
    
    std::string toString() const {
        std::stringstream ss;
        ss << "Connection #" << id << ": " 
           << sourceId << " -> " << destId 
           << ", Type: " << connectionTypeToString(type)
           << ", Bandwidth: " << bandwidth << " Mbps"
           << ", Latency: " << latency << " ms"
           << ", Reliability: " << reliability
           << ", Status: " << (isActive ? "Active" : "Inactive");
        return ss.str();
    }
};

//=============================================================================
// 2. Network Graph Structure
//=============================================================================

class Network {
private:
    std::map<int, Device> devices;
    std::map<int, Connection> connections;
    
    // Adjacency list for the network
    std::map<int, std::vector<std::pair<int, int>>> adjacencyList; // device ID -> [(dest device ID, connection ID)]
    
    int nextDeviceId = 1;
    int nextConnectionId = 1;

public:
    // Add a device to the network
    int addDevice(const std::string& name, DeviceType type, double processingCapacity) {
        int id = nextDeviceId++;
        devices.emplace(id, Device(id, name, type, processingCapacity));
        adjacencyList[id] = std::vector<std::pair<int, int>>();
        return id;
    }
    
    // Add a connection between devices
    int addConnection(int sourceId, int destId, ConnectionType type, 
                    double bandwidth, double latency, double reliability) {
        // Check that devices exist
        if (devices.find(sourceId) == devices.end() || devices.find(destId) == devices.end()) {
            throw std::runtime_error("Source or destination device does not exist");
        }
        
        int id = nextConnectionId++;
        connections.emplace(id, Connection(id, sourceId, destId, type, bandwidth, latency, reliability));
        
        // Update adjacency list (for directed graph)
        adjacencyList[sourceId].push_back(std::make_pair(destId, id));
        
        return id;
    }
    
    // Remove a device and all its connections
    void removeDevice(int deviceId) {
        if (devices.find(deviceId) == devices.end()) {
            throw std::runtime_error("Device does not exist");
        }
        
        // Remove all connections involving this device
        auto it = connections.begin();
        while (it != connections.end()) {
            if (it->second.getSourceId() == deviceId || it->second.getDestId() == deviceId) {
                // Remove from adjacency list
                int sourceId = it->second.getSourceId();
                int destId = it->second.getDestId();
                
                // Remove connection from source's adjacency list
                auto& sourceAdjList = adjacencyList[sourceId];
                sourceAdjList.erase(
                    std::remove_if(sourceAdjList.begin(), sourceAdjList.end(),
                        [destId, &it](const std::pair<int, int>& p) {
                            return p.first == destId && p.second == it->first;
                        }),
                    sourceAdjList.end()
                );
                
                // Erase the connection
                it = connections.erase(it);
            } else {
                ++it;
            }
        }
        
        // Remove device from adjacency list
        adjacencyList.erase(deviceId);
        
        // Remove the device
        devices.erase(deviceId);
    }
    
    // Remove a connection
    void removeConnection(int connectionId) {
        auto connIt = connections.find(connectionId);
        if (connIt == connections.end()) {
            throw std::runtime_error("Connection does not exist");
        }
        
        int sourceId = connIt->second.getSourceId();
        int destId = connIt->second.getDestId();
        
        // Remove from adjacency list
        auto& sourceAdjList = adjacencyList[sourceId];
        sourceAdjList.erase(
            std::remove_if(sourceAdjList.begin(), sourceAdjList.end(),
                [destId, connectionId](const std::pair<int, int>& p) {
                    return p.first == destId && p.second == connectionId;
                }),
            sourceAdjList.end()
        );
        
        // Remove the connection
        connections.erase(connectionId);
    }
    
    // Get a device by ID
    Device& getDevice(int deviceId) {
        auto it = devices.find(deviceId);
        if (it == devices.end()) {
            throw std::runtime_error("Device not found");
        }
        return it->second;
    }
    
    // Get a connection by ID
    Connection& getConnection(int connectionId) {
        auto it = connections.find(connectionId);
        if (it == connections.end()) {
            throw std::runtime_error("Connection not found");
        }
        return it->second;
    }
    
    // Get all devices
    const std::map<int, Device>& getDevices() const {
        return devices;
    }
    
    // Get all connections
    const std::map<int, Connection>& getConnections() const {
        return connections;
    }
    
    // Get the adjacency list
    const std::map<int, std::vector<std::pair<int, int>>>& getAdjacencyList() const {
        return adjacencyList;
    }
    
    // Reset all devices and connections to active
    void resetNetworkState() {
        for (auto& devicePair : devices) {
            devicePair.second.setActive(true);
        }
        
        for (auto& connectionPair : connections) {
            connectionPair.second.setActive(true);
        }
    }

    // Get neighbors of a device
    std::vector<int> getNeighbors(int deviceId) const {
        std::vector<int> neighbors;
        auto it = adjacencyList.find(deviceId);
        if (it != adjacencyList.end()) {
            for (const auto& neighbor : it->second) {
                if (devices.find(neighbor.first) != devices.end() && 
                    connections.find(neighbor.second) != connections.end() &&
                    devices.at(neighbor.first).getIsActive() &&
                    connections.at(neighbor.second).getIsActive()) {
                    neighbors.push_back(neighbor.first);
                }
            }
        }
        return neighbors;
    }

//=============================================================================
// 3. Connectivity Analysis Algorithms
//=============================================================================

    // Check if network is connected (all active devices can reach each other)
    bool isConnected() const {
        if (devices.empty()) return true;
        
        // Start from the first active device
        int startDeviceId = -1;
        for (const auto& device : devices) {
            if (device.second.getIsActive()) {
                startDeviceId = device.first;
                break;
            }
        }
        
        if (startDeviceId == -1) return true; // No active devices
        
        // BFS to find all reachable nodes
        std::set<int> visited;
        std::queue<int> queue;
        
        queue.push(startDeviceId);
        visited.insert(startDeviceId);
        
        while (!queue.empty()) {
            int currentId = queue.front();
            queue.pop();
            
            // Get all neighbors
            auto it = adjacencyList.find(currentId);
            if (it != adjacencyList.end()) {
                for (const auto& neighbor : it->second) {
                    int neighborId = neighbor.first;
                    int connectionId = neighbor.second;
                    
                    // Check if the neighbor and connection are active
                    if (devices.find(neighborId) != devices.end() && 
                        devices.at(neighborId).getIsActive() &&
                        connections.find(connectionId) != connections.end() &&
                        connections.at(connectionId).getIsActive() &&
                        visited.find(neighborId) == visited.end()) {
                        
                        queue.push(neighborId);
                        visited.insert(neighborId);
                    }
                }
            }
        }
        
        // Check if all active devices were visited
        for (const auto& device : devices) {
            if (device.second.getIsActive() && visited.find(device.first) == visited.end()) {
                return false;
            }
        }
        
        return true;
    }
    
    // Find the shortest path between two devices (using Dijkstra's algorithm)
    std::vector<int> findShortestPath(int sourceId, int destId, 
                                     std::function<double(const Connection&)> weightFunction) const {
        // Check if devices exist and are active
        if (devices.find(sourceId) == devices.end() || !devices.at(sourceId).getIsActive() ||
            devices.find(destId) == devices.end() || !devices.at(destId).getIsActive()) {
            return std::vector<int>(); // Empty path
        }
        
        // Initialize distances
        std::map<int, double> distance;
        std::map<int, int> previous;
        std::set<int> unvisited;
        
        for (const auto& device : devices) {
            if (device.second.getIsActive()) {
                distance[device.first] = std::numeric_limits<double>::infinity();
                unvisited.insert(device.first);
            }
        }
        
        distance[sourceId] = 0;
        
        while (!unvisited.empty()) {
            // Find the unvisited node with the smallest distance
            int current = -1;
            double minDistance = std::numeric_limits<double>::infinity();
            
            for (int id : unvisited) {
                if (distance[id] < minDistance) {
                    minDistance = distance[id];
                    current = id;
                }
            }
            
            if (current == -1 || current == destId) break; // No path or destination reached
            
            unvisited.erase(current);
            
            // Update distances to neighbors
            auto it = adjacencyList.find(current);
            if (it != adjacencyList.end()) {
                for (const auto& neighbor : it->second) {
                    int neighborId = neighbor.first;
                    int connectionId = neighbor.second;
                    
                    // Skip if neighbor or connection is inactive
                    if (devices.find(neighborId) == devices.end() || 
                        !devices.at(neighborId).getIsActive() ||
                        connections.find(connectionId) == connections.end() ||
                        !connections.at(connectionId).getIsActive() ||
                        unvisited.find(neighborId) == unvisited.end()) {
                        continue;
                    }
                    
                    double weight = weightFunction(connections.at(connectionId));
                    double newDistance = distance[current] + weight;
                    
                    if (newDistance < distance[neighborId]) {
                        distance[neighborId] = newDistance;
                        previous[neighborId] = current;
                    }
                }
            }
        }
        
        // Reconstruct path
        std::vector<int> path;
        if (previous.find(destId) != previous.end() || sourceId == destId) {
            for (int at = destId; at != sourceId; at = previous[at]) {
                path.push_back(at);
            }
            path.push_back(sourceId);
            std::reverse(path.begin(), path.end());
        }
        
        return path;
    }
    
    // Calculate network diameter (longest shortest path)
    double calculateDiameter(std::function<double(const Connection&)> weightFunction) const {
        double diameter = 0;
        
        for (const auto& source : devices) {
            if (!source.second.getIsActive()) continue;
            
            for (const auto& dest : devices) {
                if (!dest.second.getIsActive() || source.first == dest.first) continue;
                
                std::vector<int> path = findShortestPath(source.first, dest.first, weightFunction);
                if (!path.empty()) {
                    double pathLength = calculatePathLength(path, weightFunction);
                    diameter = std::max(diameter, pathLength);
                }
            }
        }
        
        return diameter;
    }
    
    // Calculate length of a path
    double calculatePathLength(const std::vector<int>& path, 
                             std::function<double(const Connection&)> weightFunction) const {
        double length = 0;
        
        for (size_t i = 0; i < path.size() - 1; ++i) {
            int sourceId = path[i];
            int destId = path[i + 1];
            
            // Find the connection between these devices
            for (const auto& conn : connections) {
                if (conn.second.getSourceId() == sourceId && conn.second.getDestId() == destId && 
                    conn.second.getIsActive()) {
                    length += weightFunction(conn.second);
                    break;
                }
            }
        }
        
        return length;
    }
    
    // Find all critical devices (removing them disconnects the network)
    std::vector<int> findCriticalDevices() {
        std::vector<int> criticalDevices;
        
        // Fixed: Use iterator to get non-const reference
        for (auto& devicePair : devices) {
            if (!devicePair.second.getIsActive()) continue;
            
            // Temporarily deactivate this device
            devicePair.second.setActive(false);
            
            // Check if the network is still connected
            if (!isConnected()) {
                criticalDevices.push_back(devicePair.first);
            }
            
            // Reactivate the device
            devicePair.second.setActive(true);
        }
        
        return criticalDevices;
    }
    
    // Find all critical connections (removing them disconnects the network)
    std::vector<int> findCriticalConnections() {
        std::vector<int> criticalConnections;
        
        // Fixed: Use iterator to get non-const reference
        for (auto& connectionPair : connections) {
            if (!connectionPair.second.getIsActive()) continue;
            
            // Temporarily deactivate this connection
            connectionPair.second.setActive(false);
            
            // Check if the network is still connected
            if (!isConnected()) {
                criticalConnections.push_back(connectionPair.first);
            }
            
            // Reactivate the connection
            connectionPair.second.setActive(true);
        }
        
        return criticalConnections;
    }

//=============================================================================
// 4. Flow Analysis for Bottleneck Detection
//=============================================================================

    // Ford-Fulkerson algorithm for maximum flow
    double maxFlow(int sourceId, int sinkId) {
        if (devices.find(sourceId) == devices.end() || 
            devices.find(sinkId) == devices.end() || 
            !devices.at(sourceId).getIsActive() || 
            !devices.at(sinkId).getIsActive()) {
            return 0;
        }
        
        // Create residual graph
        std::map<int, std::map<int, double>> residualCapacity;
        
        // Initialize residual capacities
        for (const auto& conn : connections) {
            if (conn.second.getIsActive()) {
                int src = conn.second.getSourceId();
                int dest = conn.second.getDestId();
                double bw = conn.second.getBandwidth();
                
                residualCapacity[src][dest] += bw;
            }
        }
        
        double maxFlow = 0;
        
        // Find augmenting paths and update residual capacities
        while (true) {
            // BFS to find an augmenting path
            std::map<int, int> parent;
            std::queue<int> queue;
            
            queue.push(sourceId);
            parent[sourceId] = -1;
            
            while (!queue.empty() && parent.find(sinkId) == parent.end()) {
                int current = queue.front();
                queue.pop();
                
                // Try all residual edges
                for (const auto& neighbor : residualCapacity[current]) {
                    int next = neighbor.first;
                    double capacity = neighbor.second;
                    
                    if (capacity > 0 && parent.find(next) == parent.end()) {
                        parent[next] = current;
                        queue.push(next);
                    }
                }
            }
            
            // If we can't reach the sink, we're done
            if (parent.find(sinkId) == parent.end()) {
                break;
            }
            
            // Find the bottleneck capacity
            double pathFlow = std::numeric_limits<double>::infinity();
            for (int v = sinkId; v != sourceId; v = parent[v]) {
                int u = parent[v];
                pathFlow = std::min(pathFlow, residualCapacity[u][v]);
            }
            
            // Update residual capacities
            for (int v = sinkId; v != sourceId; v = parent[v]) {
                int u = parent[v];
                residualCapacity[u][v] -= pathFlow;
                residualCapacity[v][u] += pathFlow; // Add reverse edge for residual capacity
            }
            
            maxFlow += pathFlow;
        }
        
        return maxFlow;
    }
    
    // Find bottlenecks in the network (connections that limit flow)
    std::vector<int> findBottlenecks(int sourceId, int sinkId) {
        std::vector<int> bottlenecks;
        
        // First calculate the max flow
        double flow = maxFlow(sourceId, sinkId);
        
        // Now check each connection to see if removing it reduces the max flow
        // Fixed: Use iterator to get non-const reference
        for (auto& connPair : connections) {
            if (!connPair.second.getIsActive()) continue;
            
            // Temporarily deactivate this connection
            connPair.second.setActive(false);
            
            // Recalculate max flow
            double newFlow = maxFlow(sourceId, sinkId);
            
            // If flow decreased, this is a bottleneck
            if (newFlow < flow) {
                bottlenecks.push_back(connPair.first);
            }
            
            // Reactivate the connection
            connPair.second.setActive(true);
        }
        
        return bottlenecks;
    }
    
    // Calculate utilization of each connection based on a flow
    std::map<int, double> calculateUtilization(int sourceId, int sinkId) {
        std::map<int, double> utilization;
        
        // First calculate the max flow
        double totalFlow = maxFlow(sourceId, sinkId);
        
        // Initialize utilization
        for (const auto& conn : connections) {
            if (conn.second.getIsActive()) {
                utilization[conn.first] = 0.0;
            }
        }
        
        // If there's no flow, return zero utilization for all connections
        if (totalFlow == 0) {
            return utilization;
        }
        
        // Create a flow network
        std::map<int, std::map<int, double>> flow;
        std::map<int, std::map<int, double>> capacity;
        
        // Initialize capacities
        for (const auto& conn : connections) {
            if (conn.second.getIsActive()) {
                int src = conn.second.getSourceId();
                int dest = conn.second.getDestId();
                double bw = conn.second.getBandwidth();
                
                capacity[src][dest] += bw;
                flow[src][dest] = 0.0;
            }
        }
        
        // Run Ford-Fulkerson to get the flow values
        std::map<int, int> parent;
        
        while (findAugmentingPath(sourceId, sinkId, capacity, flow, parent)) {
            double pathFlow = std::numeric_limits<double>::infinity();
            
            // Find bottleneck capacity
            for (int v = sinkId; v != sourceId; v = parent[v]) {
                int u = parent[v];
                pathFlow = std::min(pathFlow, capacity[u][v] - flow[u][v]);
            }
            
            // Update flow values
            for (int v = sinkId; v != sourceId; v = parent[v]) {
                int u = parent[v];
                flow[u][v] += pathFlow;
                flow[v][u] -= pathFlow; // Reverse flow for residual graph
            }
        }
        
        // Calculate utilization for each connection
        for (const auto& conn : connections) {
            if (!conn.second.getIsActive()) continue;
            
            int src = conn.second.getSourceId();
            int dest = conn.second.getDestId();
            double bw = conn.second.getBandwidth();
            
            // Calculate flow on this connection
            double connFlow = flow[src][dest];
            if (connFlow < 0) connFlow = 0; // Ignore reverse flows
            
            // Calculate utilization
            if (bw > 0) {
                utilization[conn.first] = connFlow / bw;
            }
        }
        
        return utilization;
    }
    
    // Helper for finding augmenting path in Ford-Fulkerson
    bool findAugmentingPath(int source, int sink, 
                         const std::map<int, std::map<int, double>>& capacity,
                         const std::map<int, std::map<int, double>>& flow,
                         std::map<int, int>& parent) {
        parent.clear();
        std::queue<int> queue;
        
        queue.push(source);
        parent[source] = -1;
        
        while (!queue.empty() && parent.find(sink) == parent.end()) {
            int u = queue.front();
            queue.pop();
            
            // Try all edges
            for (const auto& edge : capacity.at(u)) {
                int v = edge.first;
                double cap = edge.second;
                
                // Check if there's residual capacity
                if (parent.find(v) == parent.end() && cap > flow.at(u).at(v)) {
                    parent[v] = u;
                    queue.push(v);
                }
            }
        }
        
        return parent.find(sink) != parent.end();
    }

//=============================================================================
// 5. Failure Impact Simulation
//=============================================================================

    // Simulate a device failure
    void simulateDeviceFailure(int deviceId) {
        if (devices.find(deviceId) != devices.end()) {
            devices.at(deviceId).setActive(false);
        }
    }
    
    // Simulate a connection failure
    void simulateConnectionFailure(int connectionId) {
        if (connections.find(connectionId) != connections.end()) {
            connections.at(connectionId).setActive(false);
        }
    }
    
    // Simulate multiple device failures
    void simulateMultipleDeviceFailures(const std::vector<int>& deviceIds) {
        for (int id : deviceIds) {
            simulateDeviceFailure(id);
        }
    }
    
    // Simulate multiple connection failures
    void simulateMultipleConnectionFailures(const std::vector<int>& connectionIds) {
        for (int id : connectionIds) {
            simulateConnectionFailure(id);
        }
    }
    
    // Simulate random failures based on reliability
    void simulateRandomFailures(unsigned seed = std::random_device{}()) {
        std::mt19937 gen(seed);
        
        // Check each device for failure
        for (auto& devicePair : devices) {
            Device& device = devicePair.second;
            if (device.getIsActive()) {
                // For now, assume all devices have same reliability of 0.99
                std::bernoulli_distribution failureDist(0.01); // 1% chance of failure
                if (failureDist(gen)) {
                    device.setActive(false);
                }
            }
        }
        
        // Check each connection for failure
        for (auto& connPair : connections) {
            Connection& conn = connPair.second;
            if (conn.getIsActive()) {
                std::bernoulli_distribution failureDist(1.0 - conn.getReliability());
                if (failureDist(gen)) {
                    conn.setActive(false);
                }
            }
        }
    }
    
    // Calculate network reliability using Monte Carlo simulation
    double calculateNetworkReliability(int numSimulations = 1000) {
        int successCount = 0;
        
        for (int i = 0; i < numSimulations; ++i) {
            // Reset network
            resetNetworkState();
            
            // Simulate random failures
            simulateRandomFailures();
            
            // Check if network is still connected
            if (isConnected()) {
                successCount++;
            }
        }
        
        return static_cast<double>(successCount) / numSimulations;
    }
    
    // Calculate impact of a specific device failure
    double calculateDeviceFailureImpact(int deviceId) {
        // Save current state
        std::map<int, bool> deviceStates;
        std::map<int, bool> connectionStates;
        
        for (const auto& devicePair : devices) {
            deviceStates[devicePair.first] = devicePair.second.getIsActive();
        }
        
        for (const auto& connPair : connections) {
            connectionStates[connPair.first] = connPair.second.getIsActive();
        }
        
        // Reset network
        resetNetworkState();
        
        // Calculate baseline connectivity
        bool baselineConnected = isConnected();
        
        // Simulate failure
        simulateDeviceFailure(deviceId);
        
        // Calculate new connectivity
        bool postFailureConnected = isConnected();
        
        // Calculate impact
        double impact = 0.0;
        if (baselineConnected && !postFailureConnected) {
            impact = 1.0; // Maximum impact - network disconnected
        } else if (baselineConnected) {
            // Count how many device pairs can no longer communicate
            int totalPairs = 0;
            int disconnectedPairs = 0;
            
            for (const auto& src : devices) {
                if (!src.second.getIsActive() || src.first == deviceId) continue;
                
                for (const auto& dest : devices) {
                    if (!dest.second.getIsActive() || dest.first == deviceId || src.first == dest.first) continue;
                    
                    totalPairs++;
                    
                    std::vector<int> path = findShortestPath(
                        src.first, dest.first, 
                        [](const Connection& c) { return 1.0; }
                    );
                    
                    if (path.empty()) {
                        disconnectedPairs++;
                    }
                }
            }
            
            if (totalPairs > 0) {
                impact = static_cast<double>(disconnectedPairs) / totalPairs;
            }
        }
        
        // Restore previous state
        for (const auto& state : deviceStates) {
            devices.at(state.first).setActive(state.second);
        }
        
        for (const auto& state : connectionStates) {
            connections.at(state.first).setActive(state.second);
        }
        
        return impact;
    }

//=============================================================================
// 6. Routing Protocol Simulation
//=============================================================================

    // Simulate distance vector routing (e.g., RIP)
    std::map<int, std::map<int, std::pair<int, double>>> simulateDistanceVectorRouting() {
        // For each device, store the next hop and distance to all other devices
        // Map: sourceId -> (destId -> (nextHopId, distance))
        std::map<int, std::map<int, std::pair<int, double>>> routingTables;
        
        // Initialize routing tables
        for (const auto& src : devices) {
            if (!src.second.getIsActive()) continue;
            
            for (const auto& dest : devices) {
                if (!dest.second.getIsActive()) continue;
                
                if (src.first == dest.first) {
                    // Route to self has distance 0 and no next hop
                    routingTables[src.first][dest.first] = std::make_pair(-1, 0.0);
                } else {
                    // Set initial distance to infinity
                    routingTables[src.first][dest.first] = std::make_pair(-1, std::numeric_limits<double>::infinity());
                    
                    // Check for direct connections
                    for (const auto& neighborPair : adjacencyList.at(src.first)) {
                        int neighborId = neighborPair.first;
                        int connectionId = neighborPair.second;
                        
                        if (neighborId == dest.first && 
                            devices.at(neighborId).getIsActive() && 
                            connections.at(connectionId).getIsActive()) {
                            
                            double distance = connections.at(connectionId).getLatency();
                            routingTables[src.first][dest.first] = std::make_pair(neighborId, distance);
                            break;
                        }
                    }
                }
            }
        }
        
        // Bellman-Ford algorithm
        bool updated = true;
        int iterations = 0;
        int maxIterations = devices.size() * 2; // Prevent infinite loops
        
        while (updated && iterations < maxIterations) {
            updated = false;
            iterations++;
            
            // For each device
            for (const auto& src : devices) {
                if (!src.second.getIsActive()) continue;
                
                // For each destination
                for (const auto& dest : devices) {
                    if (!dest.second.getIsActive() || src.first == dest.first) continue;
                    
                    // Current best distance
                    double currentBestDistance = routingTables[src.first][dest.first].second;
                    
                    // Check if we can improve by going through a neighbor
                    for (const auto& neighborPair : adjacencyList.at(src.first)) {
                        int neighborId = neighborPair.first;
                        int connectionId = neighborPair.second;
                        
                        // Skip inactive neighbors or connections
                        if (!devices.at(neighborId).getIsActive() || 
                            !connections.at(connectionId).getIsActive()) {
                            continue;
                        }
                        
                        double distanceToNeighbor = connections.at(connectionId).getLatency();
                        double neighborToDestDistance = routingTables[neighborId][dest.first].second;
                        
                        double newDistance = distanceToNeighbor + neighborToDestDistance;
                        
                        if (newDistance < currentBestDistance) {
                            routingTables[src.first][dest.first] = std::make_pair(neighborId, newDistance);
                            currentBestDistance = newDistance;
                            updated = true;
                        }
                    }
                }
            }
        }
        
        return routingTables;
    }
    
    // Simulate link-state routing (e.g., OSPF)
    std::map<int, std::map<int, std::pair<int, double>>> simulateLinkStateRouting() {
        // For each device, store the next hop and distance to all other devices
        // Map: sourceId -> (destId -> (nextHopId, distance))
        std::map<int, std::map<int, std::pair<int, double>>> routingTables;
        
        // For each source device, run Dijkstra's algorithm
        for (const auto& src : devices) {
            if (!src.second.getIsActive()) continue;
            
            // Initialize distances and next hops
            std::map<int, double> distance;
            std::map<int, int> nextHop;
            std::set<int> unvisited;
            
            for (const auto& device : devices) {
                if (device.second.getIsActive()) {
                    int deviceId = device.first;
                    
                    if (deviceId == src.first) {
                        distance[deviceId] = 0.0;
                        nextHop[deviceId] = -1; // No next hop needed
                    } else {
                        distance[deviceId] = std::numeric_limits<double>::infinity();
                        nextHop[deviceId] = -1;
                        
                        // Initialize next hops for direct neighbors
                        for (const auto& neighborPair : adjacencyList.at(src.first)) {
                            int neighborId = neighborPair.first;
                            int connectionId = neighborPair.second;
                            
                            if (neighborId == deviceId && 
                                devices.at(neighborId).getIsActive() && 
                                connections.at(connectionId).getIsActive()) {
                                
                                distance[deviceId] = connections.at(connectionId).getLatency();
                                nextHop[deviceId] = neighborId;
                                break;
                            }
                        }
                    }
                    
                    unvisited.insert(deviceId);
                }
            }
            
            // Dijkstra's algorithm
            while (!unvisited.empty()) {
                // Find the unvisited node with the smallest distance
                int current = -1;
                double minDistance = std::numeric_limits<double>::infinity();
                
                for (int id : unvisited) {
                    if (distance[id] < minDistance) {
                        minDistance = distance[id];
                        current = id;
                    }
                }
                
                if (current == -1) break; // No path to remaining nodes
                
                unvisited.erase(current);
                
                // Update distances to neighbors
                for (const auto& neighborPair : adjacencyList.at(current)) {
                    int neighborId = neighborPair.first;
                    int connectionId = neighborPair.second;
                    
                    // Skip if neighbor or connection is inactive or neighbor is already visited
                    if (!devices.at(neighborId).getIsActive() || 
                        !connections.at(connectionId).getIsActive() ||
                        unvisited.find(neighborId) == unvisited.end()) {
                        continue;
                    }
                    
                    double edgeWeight = connections.at(connectionId).getLatency();
                    double newDistance = distance[current] + edgeWeight;
                    
                    if (newDistance < distance[neighborId]) {
                        distance[neighborId] = newDistance;
                        
                        // Update next hop
                        if (current == src.first) {
                            nextHop[neighborId] = neighborId;
                        } else {
                            nextHop[neighborId] = nextHop[current];
                        }
                    }
                }
            }
            
            // Build routing table for this source
            for (const auto& dest : devices) {
                if (!dest.second.getIsActive()) continue;
                
                int destId = dest.first;
                
                if (src.first == destId) {
                    routingTables[src.first][destId] = std::make_pair(-1, 0.0);
                } else {
                    routingTables[src.first][destId] = std::make_pair(
                        nextHop[destId], 
                        distance[destId]
                    );
                }
            }
        }
        
        return routingTables;
    }
    
    // Compare different routing protocols
    void compareRoutingProtocols() {
        // Get routing tables for different protocols
        auto dvRoutingTables = simulateDistanceVectorRouting();
        auto lsRoutingTables = simulateLinkStateRouting();
        
        // Calculate average path lengths for each protocol
        double dvTotalDistance = 0.0;
        double lsTotalDistance = 0.0;
        int pathCount = 0;
        
        for (const auto& src : devices) {
            if (!src.second.getIsActive()) continue;
            
            for (const auto& dest : devices) {
                if (!dest.second.getIsActive() || src.first == dest.first) continue;
                
                double dvDistance = dvRoutingTables[src.first][dest.first].second;
                double lsDistance = lsRoutingTables[src.first][dest.first].second;
                
                // Only count finite distances
                if (dvDistance < std::numeric_limits<double>::infinity() &&
                    lsDistance < std::numeric_limits<double>::infinity()) {
                    dvTotalDistance += dvDistance;
                    lsTotalDistance += lsDistance;
                    pathCount++;
                }
            }
        }
        
        double dvAverageDistance = (pathCount > 0) ? dvTotalDistance / pathCount : 0;
        double lsAverageDistance = (pathCount > 0) ? lsTotalDistance / pathCount : 0;
        
        std::cout << "Routing Protocol Comparison:" << std::endl;
        std::cout << "  Distance Vector (e.g., RIP) Average Path Length: " << dvAverageDistance << std::endl;
        std::cout << "  Link State (e.g., OSPF) Average Path Length: " << lsAverageDistance << std::endl;
        
        // Count differences in next hops
        int nextHopDifferences = 0;
        
        for (const auto& src : devices) {
            if (!src.second.getIsActive()) continue;
            
            for (const auto& dest : devices) {
                if (!dest.second.getIsActive() || src.first == dest.first) continue;
                
                int dvNextHop = dvRoutingTables[src.first][dest.first].first;
                int lsNextHop = lsRoutingTables[src.first][dest.first].first;
                
                if (dvNextHop != lsNextHop) {
                    nextHopDifferences++;
                }
            }
        }
        
        std::cout << "  Next Hop Differences: " << nextHopDifferences << std::endl;
    }

//=============================================================================
// 7. Network Topology Visualization
//=============================================================================

    // Generate a text-based visualization of the network
    void visualizeNetwork() const {
        std::cout << "Network Topology Visualization:" << std::endl;
        std::cout << "==============================" << std::endl;
        
        // Print devices
        std::cout << "Devices:" << std::endl;
        for (const auto& devicePair : devices) {
            const Device& device = devicePair.second;
            std::string statusStr = device.getIsActive() ? "Active" : "Inactive";
            std::cout << "  " << device.getId() << ": " << device.getName() 
                      << " (" << deviceTypeToString(device.getType()) << ") - " 
                      << statusStr << std::endl;
        }
        
        std::cout << std::endl;
        
        // Print connections
        std::cout << "Connections:" << std::endl;
        for (const auto& connPair : connections) {
            const Connection& conn = connPair.second;
            std::string statusStr = conn.getIsActive() ? "Active" : "Inactive";
            
            std::cout << "  " << conn.getId() << ": " 
                      << conn.getSourceId() << " -> " << conn.getDestId() 
                      << " (" << connectionTypeToString(conn.getType()) << ", " 
                      << conn.getBandwidth() << " Mbps, " 
                      << conn.getLatency() << " ms) - " 
                      << statusStr << std::endl;
        }
        
        std::cout << std::endl;
        
        // Print adjacency list
        std::cout << "Adjacency List:" << std::endl;
        for (const auto& nodeList : adjacencyList) {
            if (devices.find(nodeList.first) == devices.end() || 
                !devices.at(nodeList.first).getIsActive()) {
                continue;
            }
            
            std::cout << "  " << nodeList.first << " -> ";
            bool first = true;
            
            for (const auto& neighbor : nodeList.second) {
                if (devices.find(neighbor.first) == devices.end() || 
                    !devices.at(neighbor.first).getIsActive() ||
                    connections.find(neighbor.second) == connections.end() ||
                    !connections.at(neighbor.second).getIsActive()) {
                    continue;
                }
                
                if (!first) {
                    std::cout << ", ";
                }
                std::cout << neighbor.first;
                first = false;
            }
            
            std::cout << std::endl;
        }
    }
    
    // Generate DOT format for use with Graphviz
    std::string generateDOTFormat() const {
        std::stringstream ss;
        
        ss << "digraph Network {" << std::endl;
        ss << "  rankdir=LR;" << std::endl;
        ss << "  node [shape=box];" << std::endl;
        
        // Add nodes
        for (const auto& devicePair : devices) {
            const Device& device = devicePair.second;
            std::string color = device.getIsActive() ? "green" : "red";
            
            ss << "  " << device.getId() << " [label=\"" << device.getName() 
               << "\\n(" << deviceTypeToString(device.getType()) << ")\""
               << ", color=" << color << "];" << std::endl;
        }
        
        // Add edges
        for (const auto& connPair : connections) {
            const Connection& conn = connPair.second;
            std::string color = conn.getIsActive() ? "black" : "red";
            std::string style = conn.getIsActive() ? "solid" : "dashed";
            
            ss << "  " << conn.getSourceId() << " -> " << conn.getDestId()
               << " [label=\"" << conn.getBandwidth() << " Mbps\\n" 
               << conn.getLatency() << " ms\""
               << ", color=" << color
               << ", style=" << style << "];" << std::endl;
        }
        
        ss << "}" << std::endl;
        
        return ss.str();
    }
    
    // Export network to CSV format
    void exportToCSV(const std::string& deviceFilename, const std::string& connectionFilename) const {
        // Export devices
        std::ofstream deviceFile(deviceFilename);
        if (deviceFile.is_open()) {
            deviceFile << "ID,Name,Type,ProcessingCapacity,Active" << std::endl;
            
            for (const auto& devicePair : devices) {
                const Device& device = devicePair.second;
                deviceFile << device.getId() << ","
                           << device.getName() << ","
                           << deviceTypeToString(device.getType()) << ","
                           << device.getProcessingCapacity() << ","
                           << (device.getIsActive() ? "Yes" : "No") << std::endl;
            }
            
            deviceFile.close();
        }
        
        // Export connections
        std::ofstream connFile(connectionFilename);
        if (connFile.is_open()) {
            connFile << "ID,SourceID,DestID,Type,Bandwidth,Latency,Reliability,Active" << std::endl;
            
            for (const auto& connPair : connections) {
                const Connection& conn = connPair.second;
                connFile << conn.getId() << ","
                         << conn.getSourceId() << ","
                         << conn.getDestId() << ","
                         << connectionTypeToString(conn.getType()) << ","
                         << conn.getBandwidth() << ","
                         << conn.getLatency() << ","
                         << conn.getReliability() << ","
                         << (conn.getIsActive() ? "Yes" : "No") << std::endl;
            }
            
            connFile.close();
        }
    }

//=============================================================================
// 8. Performance Optimization Recommendations
//=============================================================================

    // Generate optimization recommendations
    std::vector<std::string> generateOptimizationRecommendations() {
        std::vector<std::string> recommendations;
        
        // Check for critical devices
        std::vector<int> criticalDevices = findCriticalDevices();
        if (!criticalDevices.empty()) {
            std::stringstream ss;
            ss << "Add redundant connections to critical devices (IDs: ";
            for (size_t i = 0; i < criticalDevices.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << criticalDevices[i];
            }
            ss << ") to prevent network partitioning.";
            recommendations.push_back(ss.str());
        }
        
        // Check for critical connections
        std::vector<int> criticalConns = findCriticalConnections();
        if (!criticalConns.empty()) {
            std::stringstream ss;
            ss << "Add redundant paths for critical connections (IDs: ";
            for (size_t i = 0; i < criticalConns.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << criticalConns[i];
            }
            ss << ") to improve network resilience.";
            recommendations.push_back(ss.str());
        }
        
        // Check for bandwidth bottlenecks
        for (const auto& src : devices) {
            if (!src.second.getIsActive()) continue;
            
            for (const auto& dest : devices) {
                if (!dest.second.getIsActive() || src.first == dest.first) continue;
                
                std::vector<int> bottlenecks = findBottlenecks(src.first, dest.first);
                if (!bottlenecks.empty()) {
                    for (int connId : bottlenecks) {
                        std::stringstream ss;
                        ss << "Increase bandwidth on connection " << connId 
                           << " (from " << connections.at(connId).getSourceId()
                           << " to " << connections.at(connId).getDestId()
                           << ") to improve flow between devices " 
                           << src.first << " and " << dest.first << ".";
                        
                        // Check if this recommendation is already in the list
                        bool isDuplicate = false;
                        for (const std::string& rec : recommendations) {
                            if (rec.find("Increase bandwidth on connection " + std::to_string(connId)) != std::string::npos) {
                                isDuplicate = true;
                                break;
                            }
                        }
                        
                        if (!isDuplicate) {
                            recommendations.push_back(ss.str());
                        }
                    }
                }
            }
        }
        
        // Check latency-based diameter
        double latencyDiameter = calculateDiameter([](const Connection& c) { return c.getLatency(); });
        if (latencyDiameter > 100.0) { // Arbitrary threshold
            recommendations.push_back("Network diameter is high (" + 
                std::to_string(latencyDiameter) + " ms). Consider adding direct connections between distant devices to reduce latency.");
        }
        
        // Check for fiber upgrade opportunities
        for (const auto& connPair : connections) {
            const Connection& conn = connPair.second;
            if (conn.getType() == ConnectionType::COPPER && conn.getBandwidth() < 1000.0) {
                std::stringstream ss;
                ss << "Consider upgrading connection " << conn.getId() 
                   << " (from " << conn.getSourceId() << " to " << conn.getDestId()
                   << ") from copper to fiber to increase bandwidth capacity.";
                recommendations.push_back(ss.str());
            }
        }
        
        // Check for load balancing opportunities
        std::map<int, int> deviceConnectionCount;
        for (const auto& connPair : connections) {
            const Connection& conn = connPair.second;
            if (conn.getIsActive()) {
                deviceConnectionCount[conn.getSourceId()]++;
                deviceConnectionCount[conn.getDestId()]++;
            }
        }
        
        for (const auto& countPair : deviceConnectionCount) {
            if (countPair.second > 5) { // Arbitrary threshold
                std::stringstream ss;
                ss << "Device " << countPair.first << " has a high number of connections (" 
                   << countPair.second << "). Consider adding load balancing to reduce strain.";
                recommendations.push_back(ss.str());
            }
        }
        
        // Check network reliability
        double reliability = calculateNetworkReliability(100); // Reduced simulation count for example
        if (reliability < 0.99) { // Arbitrary threshold
            std::stringstream ss;
            ss << "Network reliability is estimated at " << std::fixed << std::setprecision(4) 
               << reliability << ". Consider adding redundant paths to improve resilience.";
            recommendations.push_back(ss.str());
        }
        
        return recommendations;
    }
    
    // Calculate overall network health score
    double calculateNetworkHealthScore() {
        double score = 100.0; // Start with perfect score
        
        // Check connectivity (-30 if not connected)
        if (!isConnected()) {
            score -= 30.0;
        }
        
        // Check critical devices and connections (-5 for each)
        std::vector<int> criticalDevices = findCriticalDevices();
        std::vector<int> criticalConns = findCriticalConnections();
        
        score -= std::min(criticalDevices.size() * 5.0, 25.0);
        score -= std::min(criticalConns.size() * 5.0, 25.0);
        
        // Check latency diameter (-10 if above threshold)
        double latencyDiameter = calculateDiameter([](const Connection& c) { return c.getLatency(); });
        if (latencyDiameter > 100.0) {
            score -= 10.0;
        }
        
        // Check reliability (-20 if below threshold)
        double reliability = calculateNetworkReliability(50); // Reduced for example
        if (reliability < 0.99) {
            score -= 20.0 * (0.99 - reliability) / 0.99;
        }
        
        // Ensure score is between 0 and 100
        return std::max(0.0, std::min(100.0, score));
    }
};

//=============================================================================
// Main function with example usage
//=============================================================================

int main() {
    try {
        // Create a network
        Network network;
        
        // Add devices
        int router1 = network.addDevice("MainRouter", DeviceType::ROUTER, 1000.0);
        int router2 = network.addDevice("BackupRouter", DeviceType::ROUTER, 800.0);
        int switch1 = network.addDevice("Switch1", DeviceType::SWITCH, 500.0);
        int switch2 = network.addDevice("Switch2", DeviceType::SWITCH, 500.0);
        int server1 = network.addDevice("WebServer", DeviceType::SERVER, 2000.0);
        int server2 = network.addDevice("DatabaseServer", DeviceType::SERVER, 1500.0);
        int client1 = network.addDevice("Client1", DeviceType::CLIENT, 100.0);
        int client2 = network.addDevice("Client2", DeviceType::CLIENT, 100.0);
        
        // Add connections
        network.addConnection(router1, router2, ConnectionType::FIBER, 1000.0, 2.0, 0.999);
        network.addConnection(router1, switch1, ConnectionType::FIBER, 1000.0, 1.0, 0.999);
        network.addConnection(router2, switch2, ConnectionType::FIBER, 1000.0, 1.0, 0.999);
        network.addConnection(switch1, server1, ConnectionType::COPPER, 100.0, 0.5, 0.99);
        network.addConnection(switch1, client1, ConnectionType::COPPER, 100.0, 0.5, 0.99);
        network.addConnection(switch2, server2, ConnectionType::COPPER, 100.0, 0.5, 0.99);
        network.addConnection(switch2, client2, ConnectionType::COPPER, 100.0, 0.5, 0.99);
        network.addConnection(server1, server2, ConnectionType::COPPER, 100.0, 0.5, 0.95);
        
        // Display network
        std::cout << "Initial Network:" << std::endl;
        network.visualizeNetwork();
        
        // Check connectivity
        std::cout << "Network is " << (network.isConnected() ? "connected" : "disconnected") << std::endl;
        
        // Find critical devices
        std::vector<int> criticalDevices = network.findCriticalDevices();
        std::cout << "Critical devices: ";
        for (int id : criticalDevices) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
        
        // Find critical connections
        std::vector<int> criticalConns = network.findCriticalConnections();
        std::cout << "Critical connections: ";
        for (int id : criticalConns) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
        
        // Find shortest path
        std::vector<int> path = network.findShortestPath(
            client1, server2, 
            [](const Connection& c) { return c.getLatency(); }
        );
        
        std::cout << "Shortest path from Client1 to DatabaseServer: ";
        for (size_t i = 0; i < path.size(); ++i) {
            if (i > 0) std::cout << " -> ";
            std::cout << path[i];
        }
        std::cout << std::endl;
        
        // Calculate max flow
        double flow = network.maxFlow(router1, server2);
        std::cout << "Max flow from MainRouter to DatabaseServer: " << flow << " Mbps" << std::endl;
        
        // Find bottlenecks
        std::vector<int> bottlenecks = network.findBottlenecks(router1, server2);
        std::cout << "Bottlenecks between MainRouter and DatabaseServer: ";
        for (int id : bottlenecks) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
        
        // Simulate failure
        std::cout << "\nSimulating failure of MainRouter..." << std::endl;
        network.simulateDeviceFailure(router1);
        
        // Check connectivity after failure
        std::cout << "Network is " << (network.isConnected() ? "connected" : "disconnected") 
                  << " after MainRouter failure" << std::endl;
        
        // Find new path
        path = network.findShortestPath(
            client1, server2, 
            [](const Connection& c) { return c.getLatency(); }
        );
        
        std::cout << "New shortest path from Client1 to DatabaseServer: ";
        if (path.empty()) {
            std::cout << "No path available";
        } else {
            for (size_t i = 0; i < path.size(); ++i) {
                if (i > 0) std::cout << " -> ";
                std::cout << path[i];
            }
        }
        std::cout << std::endl;
        
        // Reset network
        network.resetNetworkState();
        
        // Compare routing protocols
        std::cout << "\nComparing routing protocols:" << std::endl;
        network.compareRoutingProtocols();
        
        // Generate optimization recommendations
        std::cout << "\nOptimization Recommendations:" << std::endl;
        std::vector<std::string> recommendations = network.generateOptimizationRecommendations();
        for (const std::string& rec : recommendations) {
            std::cout << "- " << rec << std::endl;
        }
        
        // Calculate network health score
        double healthScore = network.calculateNetworkHealthScore();
        std::cout << "\nNetwork Health Score: " << healthScore << "/100" << std::endl;
        
        // Export network data
        network.exportToCSV("devices.csv", "connections.csv");
        std::cout << "\nNetwork data exported to CSV files." << std::endl;
        
        // Generate DOT format
        std::string dotFormat = network.generateDOTFormat();
        std::cout << "\nGraphviz DOT Format:" << std::endl;
        std::cout << dotFormat << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}