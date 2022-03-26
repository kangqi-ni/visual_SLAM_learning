add_library(vo SHARED
    Camera.cpp
    Config.cpp
    Frame.cpp
    Map.cpp
    Mappoint.cpp
    Optimizer.cpp
    System.cpp
    Tracker.cpp
    Visualizer.cpp
    )

target_link_libraries(vo
        ${THIRD_PARTY_LIBS})