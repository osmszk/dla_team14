require 'net/https'
require 'uri'
require 'json'
require "FileUtils"
require "open-uri"

# offset = 0 #0,150,300... 150枚ずつしか取得できない

# Replace the accessKey string value with your valid access key.
@access_key = "" #FIXME

@OUT_CSV_PATH="download_imgs.#{Time.now.to_i}.csv"

def save_image(url, num, talent)
  dir_name = "./#{talent}_image/"
  FileUtils.mkdir_p(dir_name) unless FileTest.exist?(dir_name)
  file_path = "#{dir_name}/#{talent}#{num.to_s}.jpg"
  open(file_path, 'wb') do |output|
    open(url) do |data|
      output.write(data.read)
    end
  end
  return file_path
end

def download_image(term, dir, offset)
  uri  = "https://api.cognitive.microsoft.com"
  path = "/bing/v7.0/images/search"

  count = 150

  if @access_key.length != 32 then
      puts "Invalid Bing Search API subscription key!"
      puts "Please paste yours into the source code."
      abort
  end

  uri = URI(uri + path + "?q=" + URI.escape(term) + "&count=" + count.to_s + "&offset=" + offset.to_s)

  puts "Searching images for: " + term
  puts uri

  request = Net::HTTP::Get.new(uri)
  request['Ocp-Apim-Subscription-Key'] = @access_key

  response = Net::HTTP.start(uri.host, uri.port, :use_ssl => uri.scheme == 'https') do |http|
      http.request(request)
  end

  puts "\nRelevant Headers:\n\n"
  response.each_header do |key, value|
      # header names are coerced to lowercase
      if key.start_with?("bingapis-") or key.start_with?("x-msedge-") then
          puts key + ": " + value
      end
  end

  puts "\nJSON Response:\n\n"

  out_csv = []
  count.times do |i|
    begin
      image_url = JSON(response.body)["value"][i]["thumbnailUrl"]
      dist_path = save_image(image_url, i+offset, dir)
      out_csv << [term, image_url, dist_path]
      puts "#{i} saving... #{image_url}"
    rescue => e
      puts "image#{i} is error!"
      puts e
    end
  end

  puts "writing csv"

  File.open(@OUT_CSV_PATH, 'a') do |file|
    out_csv.each do |row|
      file.write(row.join(','))
    end
  end

  puts "sleeping 20sec..."
  sleep(20)
end

[0,150,300].each do |offset|
  download_image('百田夏菜子','kanako', offset)
  download_image('玉井詩織', 'shiori', offset)
  download_image('佐々木彩夏', 'ayaka', offset)
  download_image('有安杏果', 'momoka',offset)
  download_image('高城れに', 'reni',offset)
end
