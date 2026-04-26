package org.firstinspires.ftc.teamcode.opmodes.auton.test;

import com.qualcomm.robotcore.eventloop.opmode.OpMode;
import com.qualcomm.robotcore.eventloop.opmode.Autonomous;
import com.qualcomm.robotcore.hardware.DcMotor;
import com.qualcomm.robotcore.hardware.DcMotorEx;
import com.qualcomm.robotcore.hardware.VoltageSensor;
import com.qualcomm.robotcore.hardware.IMU;
import com.qualcomm.hardware.lynx.LynxModule;
import com.qualcomm.hardware.gobilda.GoBildaPinpointDriver;
import com.acmerobotics.dashboard.config.Config;
import com.bylazar.configurables.annotations.Configurable;
import org.firstinspires.ftc.robotcore.external.navigation.AngleUnit;
import org.firstinspires.ftc.robotcore.external.navigation.DistanceUnit;
import org.firstinspires.ftc.teamcode.datalogger.DataLogger;
import org.firstinspires.ftc.teamcode.util.WifiMonitor;

import java.util.ArrayList;
import java.util.List;

/**
 * ResearchLogger OpMode - Unified data collection for all noise characterization experiments
 *
 * Purpose: Logs motor commands, Pinpoint position/heading, battery voltage, and WiFi metrics at control loop rate
 * Configuration: Adjust via FTC Dashboard before each run
 *
 * Columns logged:
 *   Timestamp, LoopDeltaMs, CommandedPower, BatteryV, HeadingRad, PinpointX, PinpointY, RSSI, LinkSpeed
 *
 * NOTE: Velocity is calculated offline in Python analysis scripts via differentiation of PinpointX/Y with filtering
 */
@Config
@Configurable
@Autonomous(name = "ResearchLogger", group = "Test")
public class ResearchLogger extends OpMode {

    // ========== CONFIGURATION (FTC Dashboard) ==========
    public static double CommandedPower = 0.0;        // Power for motors (0.0 to 1.0)
    public static double CommandedPowerR = -1.0;      // If -1, use CommandedPower for both L and R; else separate right power
    public static double TestDurationSec = 30.0;      // How long to run the test
    public static String MotorNames = "FL,FR,BL,BR";        // Comma-separated motor names (e.g., "FL,FR,BL,BR")
    public static String ExperimentLabel = "test";    // Label for CSV filename
    // ========== HARDWARE ==========
    private List<DcMotorEx> motors = new ArrayList<>();
    private DcMotorEx motorL = null;           // First motor (left/primary)
    private DcMotorEx motorR = null;           // Second motor (right/secondary), if applicable
    private VoltageSensor voltageSensor;
    private IMU imu;                           // IMU for acceleration measurement
    private WifiMonitor wifiMonitor;
    private GoBildaPinpointDriver pinpoint;
    private List<LynxModule> allHubs = new ArrayList<>();

    // ========== LOGGING ==========
    private DataLogger logger;
    private long startTime;
    private long lastIterationTime;
    private long iterationCount = 0;

    @Override
    public void init() {
        telemetry.addLine("ResearchLogger initializing...");
        telemetry.addLine("Config via Dashboard:");
        telemetry.addLine("  CommandedPower: " + CommandedPower);
        telemetry.addLine("  CommandedPowerR: " + CommandedPowerR);
        telemetry.addLine("  TestDurationSec: " + TestDurationSec);
        telemetry.addLine("  MotorNames: " + MotorNames);
        telemetry.addLine("  ExperimentLabel: " + ExperimentLabel);
        telemetry.update();

        try {
            // ========== MOTOR INITIALIZATION ==========
            String[] motorNameArray = MotorNames.split(",");
            for (String name : motorNameArray) {
                name = name.trim();
                DcMotorEx motor = hardwareMap.get(DcMotorEx.class, name);
                motor.setMode(DcMotor.RunMode.RUN_WITHOUT_ENCODER);
                motor.setZeroPowerBehavior(DcMotor.ZeroPowerBehavior.FLOAT);
                motors.add(motor);
            }

            if (motors.size() > 0) motorL = motors.get(0);
            if (motors.size() > 1) motorR = motors.get(1);

            // ========== VOLTAGE SENSOR ==========
            voltageSensor = hardwareMap.voltageSensor.iterator().next();

            // ========== IMU INITIALIZATION ==========
            try {
                imu = hardwareMap.get(IMU.class, "imu");
                com.qualcomm.hardware.rev.RevHubOrientationOnRobot orientationOnRobot =
                        new com.qualcomm.hardware.rev.RevHubOrientationOnRobot(
                                com.qualcomm.hardware.rev.RevHubOrientationOnRobot.LogoFacingDirection.UP,
                                com.qualcomm.hardware.rev.RevHubOrientationOnRobot.UsbFacingDirection.FORWARD);
                imu.initialize(new com.qualcomm.robotcore.hardware.IMU.Parameters(orientationOnRobot));
            } catch (Exception e) {
                telemetry.addLine("WARNING: IMU not found, acceleration will be 0");
                imu = null;
            }

            // ========== WIFI MONITOR ==========
            wifiMonitor = new WifiMonitor();

            // ========== PINPOINT ODOMETRY ==========
            try {
                pinpoint = hardwareMap.get(GoBildaPinpointDriver.class, "pinpoint");
                pinpoint.resetPosAndIMU();  // Initialize Pinpoint position to (0, 0)
            } catch (Exception e) {
                telemetry.addLine("WARNING: Pinpoint not found, heading will be 0");
                pinpoint = null;
            }

            // ========== BULK CACHING ==========
            allHubs = hardwareMap.getAll(LynxModule.class);
            for (LynxModule hub : allHubs) {
                hub.setBulkCachingMode(LynxModule.BulkCachingMode.MANUAL);
            }

            // ========== DATA LOGGER INITIALIZATION ==========
            logger = new DataLogger("ResearchLogger_" + ExperimentLabel);
            String[] headers = {
                    "Timestamp",
                    "LoopDeltaMs",
                    "CommandedPower",
                    "BatteryV",
                    "HeadingRad",
                    "PinpointX",
                    "PinpointY",
                    "RSSI",
                    "LinkSpeed"
            };
            logger.initializeLogging(headers);

            telemetry.addLine("ResearchLogger initialized successfully");
            telemetry.addLine("Press START to begin collecting data");
            telemetry.update();

        } catch (Exception e) {
            telemetry.addLine("ERROR during init: " + e.getMessage());
            telemetry.update();
        }
    }

    @Override
    public void start() {
        startTime = System.nanoTime();
        lastIterationTime = startTime;
        logger.logComment("START: ResearchLogger test started");
        logger.logComment("  CommandedPower=" + CommandedPower);
        logger.logComment("  CommandedPowerR=" + CommandedPowerR);
        logger.logComment("  TestDurationSec=" + TestDurationSec);
        logger.logComment("  MotorNames=" + MotorNames);
        logger.logComment("  ExperimentLabel=" + ExperimentLabel);
    }

    @Override
    public void loop() {
        try {
            // ========== CLEAR BULK CACHE AT START OF LOOP ==========
            for (LynxModule hub : allHubs) {
                hub.clearBulkCache();
            }

            // ========== TIMING ==========
            long currentTime = System.nanoTime();
            double timeSec = (currentTime - startTime) / 1e9;
            double loopDeltaMs = (currentTime - lastIterationTime) / 1e6;
            lastIterationTime = currentTime;

            // ========== POWER COMMANDS ==========
            double leftPower = CommandedPower;
            double rightPower = (CommandedPowerR >= 0) ? CommandedPowerR : CommandedPower;

            // Apply power to motors
            for (int i = 0; i < motors.size(); i++) {
                double power = (i == 0) ? leftPower : rightPower;
                motors.get(i).setPower(power);
            }

            // ========== SENSOR READS ==========
            double velocityX = 0.0;
            double velocityY = 0.0;

            if (imu != null) {
                org.firstinspires.ftc.robotcore.external.navigation.YawPitchRollAngles orientation =
                        imu.getRobotYawPitchRollAngles();
            }

// Position is logged via getPosX() and getPosY() below
        // Velocity will be calculated offline via differentiation in Python analysis scripts

            double batteryV = voltageSensor.getVoltage();

            double headingRad = 0.0;
            double pinpointX = 0.0;
            double pinpointY = 0.0;
            if (pinpoint != null) {
                pinpoint.update();  // Fetch latest Pinpoint data
                headingRad = pinpoint.getHeading(AngleUnit.RADIANS);
                pinpointX = pinpoint.getPosX(DistanceUnit.MM);
                pinpointY = pinpoint.getPosY(DistanceUnit.MM);
            }

            double rssi = wifiMonitor.getSignalStrength();
            double linkSpeed = wifiMonitor.getLinkSpeed();

            // ========== LOG DATA ==========
            logger.log(
                    timeSec,
                    loopDeltaMs,
                    leftPower,
                    batteryV,
                    headingRad,
                    pinpointX,
                    pinpointY,
                    rssi,
                    linkSpeed
            );

            // ========== TELEMETRY ==========
            telemetry.addData("Time", String.format("%.2f / %.1f sec", timeSec, TestDurationSec));
            telemetry.addData("Loop Delta", String.format("%.2f ms", loopDeltaMs));
            telemetry.addData("Battery", String.format("%.2f V", batteryV));
            telemetry.addData("Position X", String.format("%.1f mm", pinpointX));
            telemetry.addData("Position Y", String.format("%.1f mm", pinpointY));
            telemetry.addData("Heading", String.format("%.3f rad", headingRad));
            telemetry.addData("Samples logged", iterationCount);
            telemetry.update();

            iterationCount++;

            // ========== TEST COMPLETE CHECK ==========
            if (timeSec >= TestDurationSec) {
                requestOpModeStop();
            }

        } catch (Exception e) {
            telemetry.addLine("ERROR in loop: " + e.getMessage());
            telemetry.update();
        }
    }

    @Override
    public void stop() {
        // ========== STOP ALL MOTORS ==========
        for (DcMotorEx motor : motors) {
            motor.setPower(0.0);
        }

        // ========== CLOSE LOGGER ==========
        logger.logComment("STOP: ResearchLogger test completed");
        logger.logComment("  Total iterations: " + iterationCount);
        logger.close();

        telemetry.addLine("ResearchLogger stopped");
        telemetry.addLine("CSV file saved to robot storage");
        telemetry.addLine("Total samples: " + iterationCount);
        telemetry.update();
    }
}