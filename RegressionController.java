package me.shinsunyoung.springbootdeveloper.controller;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;

@Controller
public class RegressionController {

    // CSV 데이터를 로드
    private List<CSVRecord> loadCsvData() throws Exception {
        Reader reader = new InputStreamReader(
                getClass().getClassLoader().getResourceAsStream("static/season.csv"), "UTF-8");
        return CSVFormat.DEFAULT.withFirstRecordAsHeader().parse(reader).getRecords();
    }

    @GetMapping("/")
    public String showForm(Model model) {
        model.addAttribute("seasons", List.of("겨울", "봄", "여름", "가을"));
        model.addAttribute("numericColumns", List.of("PM10", "SO2", "O3", "NO2", "CO", "PM25"));
        model.addAttribute("featureColumns", List.of(
                "평균기온(℃)", "최저습도(%rh)", "평균최고기온(℃)", "최고기온(℃)", "평균최저기온(℃)", "최저기온(℃)",
                "강수량(mm)", "일최다강수량(mm)", "평균풍속(m/s)", "최대풍속(m/s)", "최대풍속풍향(deg)", "최대순간풍속(m/s)",
                "최대순간풍속풍향(deg)", "일조합(hr)", "일조율(%)", "평균습도(%rh)"
        ));
        return "main";
    }

    @PostMapping("/analyze")
    public String analyze(
            @RequestParam String season,
            @RequestParam String targetColumn,
            @RequestParam String featureColumn,
            Model model) {

        try {
            List<CSVRecord> records = loadCsvData();
            List<double[]> filteredData = new ArrayList<>();
            List<Double> xData = new ArrayList<>();
            List<Double> yData = new ArrayList<>();

            // 데이터 필터링 (계절별)
            for (CSVRecord record : records) {
                int month = Integer.parseInt(record.get("월").substring(4)); // 'YYYYMM'에서 MM 추출
                String recordSeason = getSeason(month);

                if (recordSeason.equals(season)) {
                    double x = Double.parseDouble(record.get(featureColumn));
                    double y = Double.parseDouble(record.get(targetColumn));
                    filteredData.add(new double[]{x, y});
                    xData.add(x);
                    yData.add(y);
                }
            }

            // 선형 회귀 분석
            SimpleRegression regression = new SimpleRegression();
            for (double[] pair : filteredData) {
                regression.addData(pair[0], pair[1]);
            }

            double slope = regression.getSlope();
            double intercept = regression.getIntercept();

            // 평균제곱오차(MSE) 계산
            double mse = 0;
            for (double[] pair : filteredData) {
                double predicted = slope * pair[0] + intercept;
                mse += Math.pow(predicted - pair[1], 2);
            }
            mse /= filteredData.size();

            // 결과를 모델에 추가
            model.addAttribute("slope", slope);
            model.addAttribute("intercept", intercept);
            model.addAttribute("mse", mse);
            model.addAttribute("season", season);
            model.addAttribute("targetColumn", targetColumn);
            model.addAttribute("featureColumn", featureColumn);
            model.addAttribute("xData", xData);
            model.addAttribute("yData", yData);

        } catch (Exception e) {
            e.printStackTrace();
            model.addAttribute("error", "오류가 발생했습니다: " + e.getMessage());
        }

        return "result";
    }


    private String getSeason(int month) {
        if (month == 12 || month == 1 || month == 2) return "겨울";
        if (month >= 3 && month <= 5) return "봄";
        if (month >= 6 && month <= 8) return "여름";
        return "가을";
    }
}
