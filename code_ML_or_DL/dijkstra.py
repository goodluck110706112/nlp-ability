# https://zhuanlan.zhihu.com/p/63395403   这个时间复杂度不太好，但是提供了例子
# https://pythonalgos.com/dijkstras-algorithm-in-5-steps-with-python/ 这个时间复杂度很好
import heapq

def dijkstra(matrix, root):
    Inf = 1e8
    n = len(matrix)
    dist = [Inf for _ in range(n)]
    dist[root] = 0  # 到自己的距离为0
    visited = [False for _ in range(n)]
    queue = [(0, root)]  # (distance to target, target)
    while len(queue) > 0:
        _, added_target = heapq.heappop(queue)  # added_target是最短距离已知的target
        if visited[added_target]:
            continue
        # set the node to visited
        visited[added_target] = True
        # check the distance and node and distance
        for new_target, distance in enumerate(matrix[added_target]):
            # if the current node's distance + distance to the node we're visiting
            # is less than the distance of the node we're visiting on file
            # replace that distance and push the node we're visiting into the priority queue
            if dist[added_target] + distance < dist[new_target]:  # 途径added_target，到new_target的距离比较短，那就更新，也就是以后去new_target，就途径added_target
                dist[new_target] = dist[added_target] + distance
                heapq.heappush(queue, (dist[new_target], new_target))
    return dist


if __name__ == "__main__":
    inf = 1e8
    mgraph = [[0, 1, 12, inf, inf, inf],
              [inf, 0, 9, 3, inf, inf],
              [inf, inf, 0, inf, 5, inf],
              [inf, inf, 4, 0, 13, 15],
              [inf, inf, inf ,inf, 0, 4],
              [inf, inf, inf, inf ,inf, 0]]
    result = dijkstra(mgraph, 0)
    print(result)