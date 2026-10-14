// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "PipecatVisemes",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "PipecatVisemes", targets: ["PipecatVisemes"]),
    ],
    targets: [
        .target(
            name: "PipecatVisemes",
            swiftSettings: [
                // LipsyncDSP calls LAPACK's dgeev_ through Accelerate's current interface.
                .unsafeFlags(["-Xcc", "-DACCELERATE_NEW_LAPACK"]),
            ],
            linkerSettings: [.linkedFramework("Accelerate")]
        ),
    ],
    swiftLanguageModes: [.v6]
)
